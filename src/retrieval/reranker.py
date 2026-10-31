"""Clinical Reranker for PharmGuard using BAAI/bge-reranker-base Cross-Encoder.

Scores deep contextual relevance between the user's clinical query and the expanded
sentence-window context of retrieved candidates, surfacing the most pharmacologically
critical evidence for clinical reasoning.
"""

import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reranker")

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class ClinicalReranker:
    """Cross-encoder reranker for clinical sentence-window candidates."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        logger.info("Loading Clinical CrossEncoder model '%s'...", model_name)
        # CrossEncoder will automatically select cuda if available, else cpu
        self.model = CrossEncoder(model_name, device=device)
        logger.info("CrossEncoder '%s' loaded successfully.", model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        extracted_drugs: Optional[List[str]] = None,
        clinical_terms: Optional[List[str]] = None,
        top_n: int = 4,
    ) -> List[Dict[str, Any]]:
        """Score each (query, window) pair and apply Hard Entity Constraint.

        Args:
            query: The user's clinical question or medication scenario.
            candidates: List of candidate dictionaries from HybridRetriever.
            extracted_drugs: Optional list of canonical drugs in query (auto-resolved if None).
            clinical_terms: Optional list of organ/clinical condition terms (auto-detected if None).
            top_n: Number of top-ranked candidates to return (default 4).

        Returns:
            List of top_n candidates with 'rerank_score' attached.
        """
        import re
        from src.agent.entity_resolver import get_entity_resolver, COHORT_40_DRUGS

        if not candidates:
            logger.warning("Empty candidate list passed to ClinicalReranker.")
            return []

        # Auto-resolve drugs and clinical terms if not supplied
        resolver = get_entity_resolver()
        if extracted_drugs is None:
            extracted_drugs = resolver.resolve_drug_tokens(query)

        target_drugs_clean = [d.lower() for d in extracted_drugs]
        target_aliases_map = resolver.expand_drugs(target_drugs_clean)
        all_target_aliases_set = resolver.get_all_target_aliases(target_drugs_clean)

        if clinical_terms is None:
            query_lower = query.lower()
            candidate_clinical = [
                "renal", "egfr", "gfr", "kidney", "creatinine", "crcl",
                "hepatic", "liver", "cirrhosis", "dialysis", "lactic acidosis",
            ]
            clinical_terms = [t for t in candidate_clinical if t in query_lower]

        # Prepare (query, context) pairs using the expanded sentence-window
        pairs = []
        for cand in candidates:
            # Extract window from metadata or top-level key, fallback to target sentence
            window_text = (
                cand.get("metadata", {}).get("window")
                or cand.get("window")
                or cand.get("sentence", "")
            )
            pairs.append((query, window_text))

        logger.info("Scoring %d candidate pairs with CrossEncoder...", len(pairs))
        scores = self.model.predict(pairs)

        # Build list of 3rd-party foreign drugs to penalize
        foreign_drugs_check = (
            set(COHORT_40_DRUGS)
            | {
                "digoxin", "ubrogepant", "midazolam", "carbamazepine", "phenytoin",
                "clarithromycin", "rifampin", "ketoconazole", "itraconazole", "colchicine",
            }
        ) - set(target_drugs_clean) - all_target_aliases_set

        # Attach scores and apply Hard Entity Constraint
        scored_candidates = []
        for cand, raw_score in zip(candidates, scores):
            scored_cand = dict(cand)
            raw_score_float = float(raw_score)
            final_score = raw_score_float
            penalized = False

            chunk_drug = str(cand.get("metadata", {}).get("drug_name", "")).lower()
            chunk_sec = str(cand.get("metadata", {}).get("section", "")).lower()
            window_text = (
                cand.get("metadata", {}).get("window")
                or cand.get("window")
                or cand.get("sentence", "")
            )
            full_chunk_text = f"{window_text} {chunk_drug}".lower()

            # Hard Entity Constraint for multi-drug queries
            if len(target_drugs_clean) >= 2:
                # Check if covers both target drugs
                mentions_drug_0 = any(a in full_chunk_text for a in target_aliases_map.get(target_drugs_clean[0], [target_drugs_clean[0]]))
                mentions_drug_1 = any(b in full_chunk_text for b in target_aliases_map.get(target_drugs_clean[1], [target_drugs_clean[1]]))
                covers_both = mentions_drug_0 and mentions_drug_1

                # Check if covers one target drug + clinical condition
                covers_target = mentions_drug_0 or mentions_drug_1 or (chunk_drug in all_target_aliases_set)
                covers_clinical = any(term in full_chunk_text for term in clinical_terms) if clinical_terms else False
                covers_target_plus_clinical = covers_target and covers_clinical

                # Check if chunk discusses interaction with unrelated 3rd-party drug
                if chunk_sec == "drug_interactions" or "interaction" in full_chunk_text:
                    if not covers_both:
                        mentions_foreign = any(
                            re.search(rf"\b{re.escape(fd)}\b", full_chunk_text)
                            for fd in foreign_drugs_check
                        )
                        if mentions_foreign or not covers_target_plus_clinical:
                            # Hard penalty: -10.0 or cap below 0.10
                            final_score = min(raw_score_float - 10.0, 0.05)
                            penalized = True
                            logger.info(
                                "Hard Entity Constraint: Penalizing chunk '%s' (3rd-party foreign interaction, raw=%.4f -> penalized=%.4f)",
                                cand.get("chunk_id"),
                                raw_score_float,
                                final_score,
                            )

            scored_cand["rerank_score"] = final_score
            scored_cand["raw_rerank_score"] = raw_score_float
            scored_cand["penalized_foreign_interaction"] = penalized
            scored_candidates.append(scored_cand)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Multi-drug balanced selection to guarantee coverage of both requested target drugs
        if len(target_drugs_clean) >= 2 and top_n >= len(target_drugs_clean):
            slots_per_drug = max(1, top_n // len(target_drugs_clean))
            selected_candidates: List[Dict[str, Any]] = []
            selected_ids = set()

            # First priority: direct linking chunks that mention both drugs
            for cand in scored_candidates:
                if cand.get("penalized_foreign_interaction", False) or cand["rerank_score"] < 0.10:
                    continue
                full_text = (
                    cand.get("window", "")
                    + " "
                    + cand.get("sentence", "")
                    + " "
                    + str(cand.get("metadata", {}).get("drug_name", ""))
                ).lower()
                mentions_both = all(
                    any(a in full_text for a in target_aliases_map.get(d, [d]))
                    for d in target_drugs_clean
                )
                if mentions_both:
                    selected_candidates.append(cand)
                    selected_ids.add(cand.get("chunk_id"))
                    if len(selected_candidates) >= top_n:
                        break

            # Second priority: ensure each target drug has representation
            if len(selected_candidates) < top_n:
                for target_d in target_drugs_clean:
                    d_aliases = target_aliases_map.get(target_d, [target_d])
                    count_for_d = sum(
                        1 for sc in selected_candidates
                        if str(sc.get("metadata", {}).get("drug_name", "")).lower() in d_aliases
                    )
                    if count_for_d < slots_per_drug:
                        for cand in scored_candidates:
                            if cand.get("chunk_id") in selected_ids:
                                continue
                            if cand.get("penalized_foreign_interaction", False) or cand["rerank_score"] < 0.10:
                                continue
                            chunk_drug = str(cand.get("metadata", {}).get("drug_name", "")).lower()
                            if chunk_drug in d_aliases:
                                selected_candidates.append(cand)
                                selected_ids.add(cand.get("chunk_id"))
                                count_for_d += 1
                                if count_for_d >= slots_per_drug or len(selected_candidates) >= top_n:
                                    break

            # Third priority: fill remaining slots with highest scoring valid chunks
            if len(selected_candidates) < top_n:
                for cand in scored_candidates:
                    if cand.get("chunk_id") not in selected_ids:
                        selected_candidates.append(cand)
                        selected_ids.add(cand.get("chunk_id"))
                        if len(selected_candidates) >= top_n:
                            break

            top_candidates = sorted(selected_candidates[:top_n], key=lambda c: c["rerank_score"], reverse=True)
        else:
            top_candidates = scored_candidates[:top_n]

        logger.info("Reranking completed. Top %d candidates selected:", len(top_candidates))
        for rank, cand in enumerate(top_candidates, 1):
            drug = cand.get("metadata", {}).get("drug_name", "Unknown")
            sec = cand.get("metadata", {}).get("section", "Unknown")
            logger.info(
                "  [%d] Score: %.4f (Raw: %.4f, Pen: %s) | ID: %s | Drug: %s | Section: %s",
                rank,
                cand["rerank_score"],
                cand.get("raw_rerank_score", 0.0),
                cand.get("penalized_foreign_interaction", False),
                cand.get("chunk_id"),
                drug,
                sec,
            )

        return top_candidates


if __name__ == "__main__":
    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    reranker = ClinicalReranker()

    test_query = "Can a patient with moderate renal impairment taking Metformin take Paxlovid?"
    print(f"\n[PHASE 3 TEST QUERY]: '{test_query}'\n")

    candidates = retriever.retrieve(test_query, top_k=15)
    print(f"Retrieved {len(candidates)} hybrid candidates. Reranking to top 4...\n")

    top_4 = reranker.rerank(test_query, candidates, top_n=4)

    print("=" * 60)
    print("TOP 4 RERANKED CLINICAL EVIDENCE CHUNKS")
    print("=" * 60)
    for i, c in enumerate(top_4, 1):
        print(f"\nRANK #{i} | Rerank Score: {c['rerank_score']:.4f} | RRF Score: {c.get('rrf_score', 0):.4f}")
        print(f"Chunk ID: {c['chunk_id']}")
        print(f"Drug: {c['metadata'].get('drug_name')} | Section: {c['metadata'].get('section')} | Source: {c['metadata'].get('source_type')}")
        print(f"Target Sentence:\n  \"{c['sentence']}\"")
        print(f"Expanded Window:\n  \"{c['window']}\"")
        print("-" * 60)
