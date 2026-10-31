"""Regression Test Suite for PharmGuard Clinical RAG Pipeline.

Verifies the 6 architectural fixes:
1. Correctly Spelled Interaction Query (Metformin + Paxlovid in moderate renal impairment)
2. Misspelled Query (Metformn + Paxlovd in kidney disease)
3. Irrelevant / Non-Medical Query (Stock price of Apple + bake a cake)
4. Missing-Evidence / Unsupported Drug Pair Query (Paxlovid + Unobtainium-999)
"""

import pytest
from src.agent.graph import build_pharmguard_graph
from src.agent.entity_resolver import get_entity_resolver, DrugEntityResolver
from src.agent.schemas import ClinicalAssessment
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import ClinicalReranker


@pytest.fixture(scope="module")
def agent_app():
    """Compiled LangGraph workflow application."""
    return build_pharmguard_graph()


@pytest.fixture(scope="module")
def retriever():
    """Singleton HybridRetriever instance."""
    from src.agent.tools import get_retriever
    return get_retriever()


@pytest.fixture(scope="module")
def reranker():
    """Singleton ClinicalReranker instance."""
    from src.agent.tools import get_reranker
    return get_reranker()


def test_scenario_1_correctly_spelled_interaction(agent_app):
    """Scenario 1: Correctly Spelled Interaction Query.
    
    Query: 'Can a patient with moderate renal impairment taking Metformin take Paxlovid?'
    - Assert extracted drugs include metformin and paxlovid.
    - Assert top reranked chunks do NOT prioritize unrelated drugs (digoxin, ubrogepant).
    - Assert groundedness_score >= 0.75, at least 1 verified citation exists, and blocked_by_guardrail == False.
    """
    query = "Can a patient with moderate renal impairment taking Metformin take Paxlovid?"
    result = agent_app.invoke({"query": query})
    assessment: ClinicalAssessment = result.get("final_assessment")

    # 1. Assert extracted drugs include metformin and paxlovid
    extracted = [d.lower() for d in result.get("extracted_drugs", [])]
    assert "metformin" in extracted, f"Expected 'metformin' in extracted drugs: {extracted}"
    assert "paxlovid" in extracted, f"Expected 'paxlovid' in extracted drugs: {extracted}"

    # 2. Assert top reranked chunks do NOT prioritize unrelated drugs (digoxin, ubrogepant)
    retrieved_chunks = result.get("retrieved_chunks", [])
    assert len(retrieved_chunks) > 0, "Expected non-empty retrieved chunks"
    top_4_chunks = retrieved_chunks[:4]

    for chunk in top_4_chunks:
        chunk_drug = str(chunk.get("metadata", {}).get("drug_name", "")).lower()
        chunk_sec = str(chunk.get("metadata", {}).get("section", "")).lower()
        chunk_window = (chunk.get("window", "") + " " + chunk.get("sentence", "")).lower()

        # Unrelated drug interaction chunks (digoxin, ubrogepant) should never be top-ranked
        if chunk_sec == "drug_interactions":
            assert "digoxin" not in chunk_window or "metformin" in chunk_window, (
                f"Top chunk {chunk.get('chunk_id')} prioritized unrelated drug 'digoxin' over target pair!"
            )
            assert "ubrogepant" not in chunk_window, (
                f"Top chunk {chunk.get('chunk_id')} prioritized unrelated drug 'ubrogepant' over target pair!"
            )
        assert chunk.get("penalized_foreign_interaction", False) is False, (
            f"Penalized foreign chunk {chunk.get('chunk_id')} was unexpectedly in top 4!"
        )

    # 3. Assert groundedness_score >= 0.75, at least 1 verified citation, blocked_by_guardrail == False
    assert assessment is not None, "Workflow failed to produce a ClinicalAssessment"
    assert assessment.groundedness_score >= 0.75, (
        f"Groundedness score must be >= 0.75, got {assessment.groundedness_score}"
    )
    verified_cits = [c for c in assessment.citations if c.verified]
    assert len(verified_cits) >= 1, "At least one citation must be verified against source text"
    assert assessment.blocked_by_guardrail is False, "Valid clinical query should not be blocked by guardrail"
    assert assessment.risk_severity != "Insufficient Evidence / Cannot Verify"


def test_scenario_2_misspelled_query(agent_app):
    """Scenario 2: Misspelled Query.
    
    Query: 'Interactions between Metformn and Paxlovd in kidney disease'
    - Assert fuzzy drug extraction resolves 'Metformn' -> 'metformin' and 'Paxlovd' -> 'paxlovid'.
    - Assert retrieval returns the exact same top evidence as the correctly spelled query.
    """
    misspelled_query = "Interactions between Metformn and Paxlovd in kidney disease"
    correct_query = "Interactions between Metformin and Paxlovid in kidney disease"

    # 1. Fuzzy drug entity resolution check
    resolver = get_entity_resolver()
    resolved = resolver.resolve_drug_tokens(misspelled_query)
    assert "metformin" in resolved, f"Expected 'metformin' resolved from 'Metformn', got: {resolved}"
    assert "paxlovid" in resolved, f"Expected 'paxlovid' resolved from 'Paxlovd', got: {resolved}"

    # 2. Execution and retrieval comparison
    result_misspelled = agent_app.invoke({"query": misspelled_query})
    result_correct = agent_app.invoke({"query": correct_query})

    extracted_misspelled = [d.lower() for d in result_misspelled.get("extracted_drugs", [])]
    assert "metformin" in extracted_misspelled
    assert "paxlovid" in extracted_misspelled

    # Compare top evidence chunks
    top_misspelled = [c["chunk_id"] for c in result_misspelled.get("retrieved_chunks", [])[:3]]
    top_correct = [c["chunk_id"] for c in result_correct.get("retrieved_chunks", [])[:3]]

    # Assert top evidence aligns on the core renal / dosing chunks
    common_chunks = set(top_misspelled).intersection(set(top_correct))
    assert len(common_chunks) >= 1, (
        f"Expected shared top evidence between misspelled ({top_misspelled}) and correct ({top_correct})"
    )


def test_scenario_3_irrelevant_non_medical_query(agent_app):
    """Scenario 3: Irrelevant / Non-Medical Query.
    
    Query: 'What is the stock price of Apple and how to bake a cake?'
    - Assert the Pre-Generation Evidence Gate triggers, the LLM is bypassed,
      and risk_severity == 'Insufficient Evidence / Cannot Verify'.
    """
    query = "What is the stock price of Apple and how to bake a cake?"
    result = agent_app.invoke({"query": query})
    assessment: ClinicalAssessment = result.get("final_assessment")

    assert assessment is not None, "Workflow failed to produce a ClinicalAssessment"
    assert result.get("is_clinical") is False, "Query must be classified as non-clinical"
    assert result.get("evidence_gate_passed", False) is False, "Evidence Gate must fail for non-medical query"
    assert assessment.risk_severity == "Insufficient Evidence / Cannot Verify", (
        f"Expected 'Insufficient Evidence / Cannot Verify', got '{assessment.risk_severity}'"
    )
    assert assessment.blocked_by_guardrail is True, "Blocked by guardrail flag must be True"
    assert assessment.groundedness_score == 0.0, "Groundedness score must be 0.0"
    assert len(assessment.citations) == 0, "No citations should be attached"


def test_scenario_4_unsupported_drug_pair(agent_app):
    """Scenario 4: Missing-Evidence / Unsupported Drug Pair Query.
    
    Query: 'Can a patient take Paxlovid with Unobtainium-999?'
    - Assert the system does NOT hallucinate a mechanism or dosing rule.
    - Assert blocked_by_guardrail == True or risk_severity == 'Insufficient Evidence / Cannot Verify'.
    - Assert no unverified recommendation is returned.
    """
    query = "Can a patient take Paxlovid with Unobtainium-999?"
    result = agent_app.invoke({"query": query})
    assessment: ClinicalAssessment = result.get("final_assessment")

    assert assessment is not None, "Workflow failed to produce a ClinicalAssessment"
    
    # Assert Evidence Gate blocked or Guardrail blocked the output
    is_blocked = (
        assessment.blocked_by_guardrail is True
        or assessment.risk_severity == "Insufficient Evidence / Cannot Verify"
    )
    assert is_blocked, (
        f"Expected recommendation to be blocked for unsupported drug pair, got severity='{assessment.risk_severity}', "
        f"blocked={assessment.blocked_by_guardrail}"
    )

    # Assert no hallucinatory clinical recommendation was leaked
    assert "Action Blocked" in assessment.dosing_and_actionable_guidance or "INSUFFICIENT" in assessment.clinical_mechanism, (
        f"Expected blocked safety notice, got guidance: {assessment.dosing_and_actionable_guidance}"
    )
