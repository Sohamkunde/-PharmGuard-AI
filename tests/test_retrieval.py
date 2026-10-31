"""Verification and unit tests for Phase 3 Hybrid Retrieval and Cross-Encoder Reranking."""

import pytest
from src.retrieval.hybrid_retriever import HybridRetriever, compute_window_overlap
from src.retrieval.reranker import ClinicalReranker


@pytest.fixture(scope="module")
def retriever():
    """Shared fixture for HybridRetriever instance."""
    return HybridRetriever()


@pytest.fixture(scope="module")
def reranker():
    """Shared fixture for ClinicalReranker instance."""
    return ClinicalReranker()


def test_window_overlap_calculation():
    """Test Jaccard window overlap logic."""
    text1 = "Patient on metformin must monitor eGFR regularly. Dose reduction is required."
    text2 = "Patient on metformin must monitor eGFR regularly. Dose reduction is required. Lactic acidosis is rare."
    overlap = compute_window_overlap(text1, text2)
    assert overlap > 0.60

    text3 = "Amiodarone interacts with Paxlovid causing fatal cardiac arrhythmias."
    no_overlap = compute_window_overlap(text1, text3)
    assert no_overlap < 0.20


def test_hybrid_retriever_initialization(retriever):
    """Test loading of both ChromaDB collection and BM25 index."""
    assert retriever.corpus_size == 6245
    assert len(retriever.chunk_ids) == 6245
    assert len(retriever.sentences) == 6245


def test_dense_only_retrieve(retriever):
    """Test dense-only retrieval baseline for RAGAS evaluation benchmark."""
    query = "Apixaban bleeding risk with aspirin"
    results = retriever.dense_only_retrieve(query, top_k=5)
    assert len(results) == 5
    for r in results:
        assert "chunk_id" in r
        assert "sentence" in r
        assert "window" in r
        assert "metadata" in r
        assert r["retrieval_mode"] == "dense_only"


def test_hybrid_retrieve_and_dedup(retriever):
    """Test hybrid retrieval and sentence-window deduplication."""
    query = "Colchicine contraindications in renal impairment with clarithromycin"
    results = retriever.retrieve(query, top_k=15)
    assert 0 < len(results) <= 15

    for r in results:
        assert "rrf_score" in r
        assert r["rrf_score"] > 0
        assert "metadata" in r
        assert "window" in r

    # Verify deduplication: no two chunks from same drug+section have window overlap >= 0.60
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            drug_i = results[i]["metadata"].get("drug_name")
            drug_j = results[j]["metadata"].get("drug_name")
            sec_i = results[i]["metadata"].get("section")
            sec_j = results[j]["metadata"].get("section")
            if drug_i == drug_j and sec_i == sec_j:
                overlap = compute_window_overlap(results[i]["window"], results[j]["window"])
                assert overlap < 0.60, f"Overlapping windows not deduplicated between {results[i]['chunk_id']} and {results[j]['chunk_id']}"


def test_metformin_paxlovid_clinical_query(retriever, reranker):
    """Specific Phase 3 verification test query:
    'Can a patient with moderate renal impairment taking Metformin take Paxlovid?'
    """
    test_query = "Can a patient with moderate renal impairment taking Metformin take Paxlovid?"
    print("\n" + "=" * 70)
    print(f"CLINICAL TEST QUERY: {test_query}")
    print("=" * 70)

    # 1. Retrieve candidates
    candidates = retriever.retrieve(test_query, top_k=15)
    assert len(candidates) > 0

    # 2. Rerank candidates to top 4
    top_4 = reranker.rerank(test_query, candidates, top_n=4)
    assert len(top_4) == 4

    # 3. Verify descending score order
    scores = [c["rerank_score"] for c in top_4]
    assert scores == sorted(scores, reverse=True), "Reranked scores are not strictly descending"

    # 4. Print results to terminal as requested (safe for Windows cp1252 stdout)
    def safe_text(s: str) -> str:
        return s.replace("\u2265", ">=").replace("\u2264", "<=").encode("ascii", errors="replace").decode("ascii")

    print("\nTOP 4 RERANKED CLINICAL EVIDENCE CHUNKS:")
    for rank, cand in enumerate(top_4, 1):
        meta = cand.get("metadata", {})
        clean_sentence_text = safe_text(cand['sentence'])
        clean_window_text = safe_text(cand['window'][:200])
        print(f"\n[{rank}] Score: {cand['rerank_score']:.4f} | RRF Score: {cand.get('rrf_score', 0):.4f}")
        print(f"    Chunk ID:    {cand['chunk_id']}")
        print(f"    Drug/Topic:  {meta.get('drug_name')}")
        print(f"    Section:     {meta.get('section')}")
        print(f"    Source Type: {meta.get('source_type')}")
        print(f"    Sentence:    {clean_sentence_text}")
        print(f"    Window:      {clean_window_text}...")
    print("=" * 70 + "\n")

    # 5. Clinical assertion: top retrieved chunks must address Metformin, Paxlovid, or Renal impairment
    retrieved_drugs = [c["metadata"].get("drug_name") for c in top_4]
    assert any(
        d in ["Metformin", "Paxlovid", "Renal Impairment & eGFR Dosing", "CYP3A4 Drug-Drug Interactions"]
        for d in retrieved_drugs
    ), f"Expected relevant drugs/guidelines in top 4, got: {retrieved_drugs}"
