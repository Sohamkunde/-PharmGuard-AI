"""Phase 4 LangGraph Agentic Workflow & Pydantic Guardrail Tests."""

import pytest
from src.agent.graph import build_pharmguard_graph
from src.agent.schemas import ClinicalAssessment


@pytest.fixture(scope="module")
def agent_app():
    """Compiled LangGraph workflow application."""
    return build_pharmguard_graph()


def test_query_1_metformin_paxlovid_renal(agent_app):
    """Test Query 1: Moderate renal impairment + Metformin + Paxlovid."""
    query = "Can a patient with moderate renal impairment taking Metformin take Paxlovid?"
    print("\n" + "=" * 70)
    print(f"[LANGGRAPH QUERY 1]: {query}")
    print("=" * 70)

    result = agent_app.invoke({"query": query})
    assessment: ClinicalAssessment = result.get("final_assessment")

    assert assessment is not None, "Workflow failed to produce a ClinicalAssessment"
    assert assessment.risk_severity in [
        "Major / Dose Adjustment Required",
        "Critical / Contraindicated",
        "Moderate / Monitor",
    ]
    assert assessment.groundedness_score > 0.0, "Groundedness score must be positive"
    assert len(assessment.citations) > 0, "At least one citation must be present"
    assert any(c.verified for c in assessment.citations), "At least one citation must be verified"

    # Verify tool selection
    selected_tools = result.get("selected_tools", [])
    assert "search_fda_and_guidelines" in selected_tools
    assert "renal_egfr_calculator_tool" in selected_tools

    # Safe print markdown summary
    safe_md = (
        assessment.to_markdown_summary()
        .replace("\u2265", ">=")
        .replace("\u2264", "<=")
        .encode("ascii", errors="replace")
        .decode("ascii")
    )
    print(safe_md)
    print("=" * 70 + "\n")


def test_query_2_fluoxetine_phenelzine_washout(agent_app):
    """Test Query 2: Fluoxetine + Phenelzine (SSRI + MAOI Contraindication)."""
    query = (
        "Patient is taking Fluoxetine 40mg daily and the doctor wants to start Phenelzine "
        "for refractory depression. What is the risk and required protocol?"
    )
    print("\n" + "=" * 70)
    print(f"[LANGGRAPH QUERY 2]: {query}")
    print("=" * 70)

    result = agent_app.invoke({"query": query})
    assessment: ClinicalAssessment = result.get("final_assessment")

    assert assessment is not None, "Workflow failed to produce a ClinicalAssessment"
    assert assessment.risk_severity == "Critical / Contraindicated"
    guidance_norm = (
        assessment.dosing_and_actionable_guidance.lower()
        .replace("\u2011", "-")
        .replace("\u202f", " ")
        .replace("wash-out", "washout")
    )
    assert "5-week" in guidance_norm or "5 week" in guidance_norm or "washout" in guidance_norm
    assert assessment.groundedness_score >= 0.6, "Groundedness score must be >= 0.6"

    safe_md = (
        assessment.to_markdown_summary()
        .replace("\u2265", ">=")
        .replace("\u2264", "<=")
        .encode("ascii", errors="replace")
        .decode("ascii")
    )
    print(safe_md)
    print("=" * 70 + "\n")


def test_query_3_out_of_domain_messi(agent_app):
    """Test Query 3: Out-of-Domain Non-Clinical Query ('who is messi')."""
    query = "who is messi"
    print("\n" + "=" * 70)
    print(f"[LANGGRAPH QUERY 3 - OUT OF DOMAIN]: {query}")
    print("=" * 70)

    result = agent_app.invoke({"query": query})
    assessment: ClinicalAssessment = result.get("final_assessment")

    assert assessment is not None, "Workflow failed to produce a ClinicalAssessment"
    assert result.get("is_clinical") is False, "Query must be classified as non-clinical"
    assert assessment.risk_severity == "Low / No Known Conflict"
    assert len(assessment.citations) == 0, "Non-clinical query must have 0 citations"
    assert "Out-of-Domain" in assessment.clinical_mechanism

    print(f"Risk Severity: {assessment.risk_severity}")
    print(f"Mechanism Note: {assessment.clinical_mechanism}")
    print("=" * 70 + "\n")


def test_query_4_out_of_domain_kunuuuuu(agent_app):
    """Test Query 4: Conversational personal name statement ('my name is kunuuuuu')."""
    query = "my name is kunuuuuu"
    print("\n" + "=" * 70)
    print(f"[LANGGRAPH QUERY 4 - CONVERSATIONAL / OOD]: {query}")
    print("=" * 70)

    result = agent_app.invoke({"query": query})
    assessment: ClinicalAssessment = result.get("final_assessment")

    assert assessment is not None, "Workflow failed to produce a ClinicalAssessment"
    assert result.get("is_clinical") is False, "Query must be classified as non-clinical"
    assert assessment.risk_severity == "Low / No Known Conflict"
    assert len(assessment.citations) == 0, "Non-clinical query must have 0 citations"
    assert "Out-of-Domain" in assessment.clinical_mechanism

    print(f"Risk Severity: {assessment.risk_severity}")
    print(f"Mechanism Note: {assessment.clinical_mechanism}")
    print("=" * 70 + "\n")


def test_query_5_single_drug_inquiry(agent_app):
    """Test Query 5: Single-drug food inquiry (safe, no false contraindication)."""
    query = "Can a patient take Metformin 500mg with breakfast?"
    print("\n" + "=" * 70)
    print(f"[LANGGRAPH QUERY 5 - SINGLE DRUG / FOOD INQUIRY]: {query}")
    print("=" * 70)

    result = agent_app.invoke({"query": query})
    assessment: ClinicalAssessment = result.get("final_assessment")

    assert assessment is not None, "Workflow failed to produce a ClinicalAssessment"
    assert result.get("is_clinical") is True, "Query must be classified as clinical"
    assert assessment.risk_severity in ["Low / No Known Conflict", "Moderate / Monitor"], (
        f"Single drug food inquiry should NOT be marked Critical: got {assessment.risk_severity}"
    )

    print(f"Risk Severity: {assessment.risk_severity}")
    print(f"Mechanism: {assessment.clinical_mechanism[:120]}...")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    app = build_pharmguard_graph()
    test_query_1_metformin_paxlovid_renal(app)
    test_query_2_fluoxetine_phenelzine_washout(app)
    test_query_3_out_of_domain_messi(app)
    test_query_4_out_of_domain_kunuuuuu(app)
    test_query_5_single_drug_inquiry(app)
