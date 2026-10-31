"""Hybrid Retriever for PharmGuard: Dense ChromaDB + BM25 Lexical + Reciprocal Rank Fusion.

Combines semantic dense retrieval (ChromaDB + BAAI/bge-base-en-v1.5) and lexical
sparse retrieval (rank_bm25 BM25Okapi) using Reciprocal Rank Fusion (RRF, k=60).
Deduplicates overlapping sentence-window contexts to prevent downstream LLM redundancy.
"""

import re
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import chromadb
from chromadb.utils import embedding_functions

from src.indexing.store import (
    tokenize_bm25,
    get_embedding_function,
    CHROMA_DIR,
    BM25_INDEX_FILE,
    COLLECTION_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hybrid_retriever")


def compute_window_overlap(text1: str, text2: str) -> float:
    """Calculate word-level Jaccard similarity between two window strings."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0


class HybridRetriever:
    """Hybrid Clinical Retriever combining dense vector search and BM25 with RRF."""

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        bm25_file: Path = BM25_INDEX_FILE,
        collection_name: str = COLLECTION_NAME,
        embedding_func = None,
    ):
        self.chroma_dir = Path(chroma_dir)
        self.bm25_file = Path(bm25_file)
        self.collection_name = collection_name

        logger.info("Initializing HybridRetriever...")
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.embedding_func = embedding_func or get_embedding_function()
        self.collection = self.client.get_collection(
            name=self.collection_name,
            embedding_function=self.embedding_func,
        )

        logger.info("Loading BM25 index payload from %s...", self.bm25_file)
        with open(self.bm25_file, "rb") as f:
            self.bm25_payload = pickle.load(f)

        self.bm25 = self.bm25_payload["bm25"]
        self.chunk_ids = self.bm25_payload["chunk_ids"]
        self.sentences = self.bm25_payload["sentences"]
        self.windows = self.bm25_payload["windows"]
        self.metadatas = self.bm25_payload["metadatas"]
        self.corpus_size = self.bm25_payload["corpus_size"]

        # Fast in-memory lookup by chunk_id
        self.id_to_index = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        logger.info("HybridRetriever initialized successfully (%d documents).", self.corpus_size)

    def dense_only_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top_k documents using only ChromaDB dense vector search.
        
        Used for baseline comparison against hybrid retrieval in RAGAS evaluations.
        """
        results = self.collection.query(query_texts=[query], n_results=top_k)
        formatted = []

        if not results or not results["ids"] or not results["ids"][0]:
            return formatted

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if results.get("distances") and results["distances"] else [0.0] * len(ids)

        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
            window = meta.get("window", doc)
            formatted.append({
                "chunk_id": cid,
                "sentence": doc,
                "window": window,
                "metadata": meta,
                "dense_rank": rank,
                "dense_distance": float(dist),
                # Cosine similarity approximation from cosine distance
                "score": float(1.0 - dist) if dist is not None else 0.0,
                "retrieval_mode": "dense_only",
            })

        return formatted

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        target_drugs: Optional[List[str]] = None,
        target_aliases: Optional[Dict[str, List[str]]] = None,
        clinical_terms: Optional[List[str]] = None,
        dense_candidates_count: int = 25,
        bm25_candidates_count: int = 25,
        rrf_k: int = 60,
        dedup_overlap_threshold: float = 0.60,
    ) -> List[Dict[str, Any]]:
        """Perform Hybrid Retrieval combining Dense + BM25 via Reciprocal Rank Fusion (RRF).

        Enhanced with:
        - Metadata Pre-Filtering: When target drugs D are extracted, restricts candidate
          searches primarily to chunks where metadata['drug_name'] is in D or source_type == 'clinical_guideline'.
        - 3-Tier RRF Scoring:
          * Tier 1 (Direct Pair Match): 3.0x multiplier for chunks linking target drugs.
          * Tier 2 (Organ/Dosing Condition Match): 2.0x multiplier for dosing/warnings matching queried clinical parameters.
          * Tier 3 (Foreign Drug Penalty): 0.15x penalty for interaction chunks discussing unrelated third drugs.

        Args:
            query: Clinical query string.
            top_k: Number of deduplicated, reranker-ready chunks to return.
            target_drugs: Optional list of canonical drugs in query (auto-resolved if None).
            target_aliases: Optional dict of drug -> aliases (auto-expanded if None).
            clinical_terms: Optional list of organ/clinical condition terms (auto-detected if None).
            dense_candidates_count: Number of results pulled from ChromaDB.
            bm25_candidates_count: Number of results pulled from BM25.
            rrf_k: Smoothing constant for RRF.
            dedup_overlap_threshold: Jaccard threshold for dropping overlapping windows.

        Returns:
            List of ranked candidate dicts with combined RRF scores and deduplicated windows.
        """
        from src.agent.entity_resolver import get_entity_resolver, COHORT_40_DRUGS

        # Auto-resolve drugs and aliases if not explicitly provided
        resolver = get_entity_resolver()
        if target_drugs is None:
            target_drugs = resolver.resolve_drug_tokens(query)
        
        target_drugs_clean = [d.lower() for d in target_drugs]
        
        if target_aliases is None:
            target_aliases = resolver.expand_drugs(target_drugs_clean)

        # Detect clinical terms in query if not provided
        if clinical_terms is None:
            query_lower = query.lower()
            candidate_clinical = [
                "renal", "egfr", "gfr", "kidney", "creatinine", "crcl",
                "hepatic", "liver", "cirrhosis", "dialysis", "lactic acidosis",
            ]
            clinical_terms = [t for t in candidate_clinical if t in query_lower]

        all_target_aliases_set = set()
        for d, aliases in target_aliases.items():
            for a in aliases:
                all_target_aliases_set.add(a.lower())

        # 1. Query Dense ChromaDB with optional metadata pre-filtering
        dense_results = None
        if target_drugs_clean:
            # Build casing variants for ChromaDB where clause
            drug_name_variants = []
            for d in target_drugs_clean:
                drug_name_variants.extend([d.title(), d.capitalize(), d.lower(), d.upper()])
            
            where_filter = {
                "$or": [
                    {"drug_name": {"$in": drug_name_variants}},
                    {"source_type": "clinical_guideline"},
                ]
            }
            try:
                dense_results = self.collection.query(
                    query_texts=[query],
                    n_results=dense_candidates_count,
                    where=where_filter,
                )
            except Exception as exc:
                logger.warning("ChromaDB pre-filtered query failed (%s), falling back to unconstrained query.", exc)
                dense_results = None

        if not dense_results or not dense_results.get("ids") or not dense_results["ids"][0]:
            dense_results = self.collection.query(
                query_texts=[query],
                n_results=dense_candidates_count,
            )

        dense_ranked_ids: List[str] = []
        dense_scores_map: Dict[str, float] = {}
        if dense_results and dense_results["ids"] and dense_results["ids"][0]:
            dense_ranked_ids = dense_results["ids"][0]
            if dense_results.get("distances") and dense_results["distances"][0]:
                for cid, dist in zip(dense_ranked_ids, dense_results["distances"][0]):
                    dense_scores_map[cid] = float(1.0 - dist)

        # 2. Query Sparse BM25 with Metadata Pre-Filtering
        query_tokens = tokenize_bm25(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Filter indices primarily to target drugs and clinical guidelines if target drugs exist
        candidate_indices = list(range(len(bm25_scores)))
        if target_drugs_clean:
            filtered_indices = [
                i for i in candidate_indices
                if str(self.metadatas[i].get("drug_name", "")).lower() in target_drugs_clean
                or self.metadatas[i].get("source_type") == "clinical_guideline"
            ]
            if len(filtered_indices) >= 5:
                candidate_indices = filtered_indices

        top_bm25_indices = sorted(
            candidate_indices,
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:bm25_candidates_count]

        bm25_ranked_ids: List[str] = [self.chunk_ids[i] for i in top_bm25_indices]
        bm25_scores_map: Dict[str, float] = {
            self.chunk_ids[i]: float(bm25_scores[i]) for i in top_bm25_indices
        }

        # 3. Reciprocal Rank Fusion (RRF) with Co-Occurrence & Tier Boosting
        all_candidate_ids: Set[str] = set(dense_ranked_ids).union(set(bm25_ranked_ids))
        fused_candidates: List[Dict[str, Any]] = []

        # Identify other unrelated cohort drugs for foreign drug penalty
        unrelated_cohort_drugs = [
            d for d in COHORT_40_DRUGS if d.lower() not in target_drugs_clean
        ]
        # Include other common interaction drugs that trigger false positives
        unrelated_other_drugs = [
            "digoxin", "ubrogepant", "midazolam", "carbamazepine", "phenytoin",
            "clarithromycin", "rifampin", "ketoconazole", "itraconazole", "colchicine",
        ]
        foreign_drugs_check = set(unrelated_cohort_drugs + unrelated_other_drugs) - set(target_drugs_clean) - all_target_aliases_set

        for cid in all_candidate_ids:
            idx = self.id_to_index.get(cid)
            if idx is None:
                continue

            sentence = self.sentences[idx]
            window = self.windows[idx]
            metadata = self.metadatas[idx]

            dense_rank = dense_ranked_ids.index(cid) + 1 if cid in dense_ranked_ids else None
            bm25_rank = bm25_ranked_ids.index(cid) + 1 if cid in bm25_ranked_ids else None

            base_rrf = 0.0
            if dense_rank:
                base_rrf += 1.0 / (rrf_k + dense_rank)
            if bm25_rank:
                base_rrf += 1.0 / (rrf_k + bm25_rank)

            # Categorize into Tiers
            tier = "Standard"
            multiplier = 1.0

            chunk_drug = str(metadata.get("drug_name", "")).lower()
            chunk_section = str(metadata.get("section", "")).lower()
            chunk_source = str(metadata.get("source_type", "")).lower()
            chunk_text = (window + " " + sentence).lower()

            # Tier 1 (Direct Pair Match): Multiplier 3.0x
            if len(target_drugs_clean) >= 2:
                # Check if chunk mentions both drugs, or from drug A mentioning drug B
                mentions_drug_a = any(a in chunk_text for a in target_aliases.get(target_drugs_clean[0], [target_drugs_clean[0]]))
                mentions_drug_b = any(b in chunk_text for b in target_aliases.get(target_drugs_clean[1], [target_drugs_clean[1]]))

                # Direct match: Drug A's label mentioning Drug B, or vice-versa, or guideline mentioning both
                if (chunk_drug in target_aliases.get(target_drugs_clean[0], []) and mentions_drug_b) or \
                   (chunk_drug in target_aliases.get(target_drugs_clean[1], []) and mentions_drug_a) or \
                   (chunk_source == "clinical_guideline" and mentions_drug_a and mentions_drug_b):
                    multiplier = 3.0
                    tier = "Tier 1: Direct Pair Match"

            # Tier 2 (Organ/Dosing Condition Match): Multiplier 2.0x
            if tier == "Standard" and clinical_terms:
                is_target_drug_chunk = (chunk_drug in target_drugs_clean) or (chunk_drug in all_target_aliases_set)
                is_dosing_or_warning = chunk_section in ("dosage_and_administration", "warnings_and_cautions", "boxed_warning")
                mentions_clinical = any(term in chunk_text for term in clinical_terms)

                if is_target_drug_chunk and is_dosing_or_warning and mentions_clinical:
                    multiplier = 2.0
                    tier = "Tier 2: Organ/Dosing Condition Match"

            # Tier 3 (Foreign Drug Penalty): Multiplier 0.15x
            if tier == "Standard" and len(target_drugs_clean) >= 2:
                if chunk_section == "drug_interactions" and (chunk_drug in target_drugs_clean or chunk_drug in all_target_aliases_set):
                    # Check if it does NOT mention the other target drug
                    other_targets = [d for d in target_drugs_clean if d != chunk_drug]
                    other_aliases = [a for ot in other_targets for a in target_aliases.get(ot, [ot])]
                    mentions_other = any(oa in chunk_text for oa in other_aliases)
                    
                    if not mentions_other:
                        # Check if it mentions unrelated 3rd-party drugs
                        mentions_foreign = any(
                            re.search(rf"\b{re.escape(fd)}\b", chunk_text)
                            for fd in foreign_drugs_check
                        )
                        if mentions_foreign:
                            multiplier = 0.15
                            tier = "Tier 3: Foreign Drug Penalty"

            final_rrf = base_rrf * multiplier

            fused_candidates.append({
                "chunk_id": cid,
                "sentence": sentence,
                "window": window,
                "metadata": metadata,
                "rrf_score": final_rrf,
                "base_rrf_score": base_rrf,
                "tier": tier,
                "multiplier": multiplier,
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "dense_score": dense_scores_map.get(cid),
                "bm25_score": bm25_scores_map.get(cid, 0.0),
                "retrieval_mode": "hybrid_rrf",
            })

        # Sort by RRF score descending
        fused_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)

        # 4. Deduplicate Overlapping Sentence Windows
        deduplicated: List[Dict[str, Any]] = []
        for candidate in fused_candidates:
            cand_window = candidate.get("window", "")
            cand_drug = candidate.get("metadata", {}).get("drug_name", "")
            cand_sec = candidate.get("metadata", {}).get("section", "")

            is_duplicate = False
            for accepted in deduplicated:
                acc_drug = accepted.get("metadata", {}).get("drug_name", "")
                acc_sec = accepted.get("metadata", {}).get("section", "")

                # Only check overlap within same drug and section to avoid false matches
                if cand_drug == acc_drug and cand_sec == acc_sec:
                    acc_window = accepted.get("window", "")
                    overlap = compute_window_overlap(cand_window, acc_window)
                    if overlap >= dedup_overlap_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                deduplicated.append(candidate)

            if len(deduplicated) >= top_k:
                break

        logger.info(
            "Hybrid retrieve: query='%s...' -> %d raw candidates -> %d deduplicated (returned top %d)",
            query[:40],
            len(fused_candidates),
            len(deduplicated),
            min(top_k, len(deduplicated)),
        )
        return deduplicated[:top_k]


if __name__ == "__main__":
    retriever = HybridRetriever()
    test_q = "Can a patient with moderate renal impairment taking Metformin take Paxlovid?"
    print(f"\nTesting Hybrid Retriever for query: '{test_q}'\n")
    results = retriever.retrieve(test_q, top_k=5)
    for idx, r in enumerate(results, 1):
        print(f"[{idx}] {r['chunk_id']} (RRF: {r['rrf_score']:.4f}, Dense Rank: {r['dense_rank']}, BM25 Rank: {r['bm25_rank']})")
        print(f"    Drug: {r['metadata'].get('drug_name')} | Section: {r['metadata'].get('section')}")
        print(f"    Sentence: {r['sentence'][:100]}...\n")
