"""Regression Tests for Citation Verification Diagnostics, Metadata Auditing,
Claim Grounding Separation, and System-Diagnostic Routing.

Fulfills Requirements 1-6:
1. Verified quotations (exact & normalized match)
2. Mismatched quotations (modified words & paraphrasing with specific failure reason)
3. Incorrect source metadata (section/source mismatch detected and reported)
4. Unsupported claims with valid quotations (verbatim quotation matches source, but claim_support_status='unsupported')
5. System-diagnostic questions ("Why is the Metformin citation marked UNVERIFIED CITATION?") routed to diagnostic path without triggering Evidence Gate, explaining root cause, and redacting secrets.
6. Evidence Gate preservation for clinical queries.
"""

import re
import pytest
from src.agent.schemas import Citation, ClinicalAssessment, CitationVerificationDiagnostic
from src.agent.guardrails import verify_citation_detailed, verify_citation
from src.agent.graph import build_pharmguard_graph


@pytest.fixture(scope="module")
def agent_app():
    """Compiled LangGraph workflow application."""
    return build_pharmguard_graph()


@pytest.fixture
def sample_metformin_chunk():
    """Real Metformin contraindications chunk representation."""
    return {
        "chunk_id": "fda_label_Metformin_contraindications_0000",
        "sentence": "Metformin hydrochloride tablets are contraindicated in patients with: Severe renal impairment (eGFR below 30 mL/min/1.73 m 2 )",
        "window": (
            "4 CONTRAINDICATIONS Metformin hydrochloride tablets are contraindicated in patients with: "
            "Severe renal impairment (eGFR below 30 mL/min/1.73 m 2 ) Known hypersensitivity to metformin hydrochloride "
            "Acute or chronic metabolic acidosis, including lactic acidosis, with or without coma."
        ),
        "metadata": {
            "drug_name": "Metformin",
            "section": "contraindications",
            "source_type": "fda_label",
            "window": (
                "4 CONTRAINDICATIONS Metformin hydrochloride tablets are contraindicated in patients with: "
                "Severe renal impairment (eGFR below 30 mL/min/1.73 m 2 ) Known hypersensitivity to metformin hydrochloride "
                "Acute or chronic metabolic acidosis, including lactic acidosis, with or without coma."
            ),
        },
        "rerank_score": 0.95,
    }


@pytest.fixture
def sample_paxlovid_chunk():
    """Real Paxlovid dosage chunk representation."""
    return {
        "chunk_id": "fda_label_Paxlovid_dosage_and_administration_0029",
        "sentence": "In patients with moderate renal impairment (eGFR >= 30 to < 60 mL/min), the dosage of PAXLOVID is 150 mg nirmatrelvir and 100 mg ritonavir twice daily for 5 days.",
        "window": (
            "2 DOSAGE AND ADMINISTRATION 2.3 Dosage in Patients with Renal Impairment "
            "In patients with moderate renal impairment (eGFR >= 30 to < 60 mL/min), the dosage of PAXLOVID "
            "is 150 mg nirmatrelvir and 100 mg ritonavir twice daily for 5 days."
        ),
        "metadata": {
            "drug_name": "Paxlovid",
            "section": "dosage_and_administration",
            "source_type": "fda_label",
            "window": (
                "2 DOSAGE AND ADMINISTRATION 2.3 Dosage in Patients with Renal Impairment "
                "In patients with moderate renal impairment (eGFR >= 30 to < 60 mL/min), the dosage of PAXLOVID "
                "is 150 mg nirmatrelvir and 100 mg ritonavir twice daily for 5 days."
            ),
        },
        "rerank_score": 0.98,
    }


# -----------------------------------------------------------------------------
# Test 1: Verified Quotations (Exact and Normalized Match)
# -----------------------------------------------------------------------------
def test_1_verified_quotation(sample_metformin_chunk):
    """Assert verbatim quote in matching section produces verification_result='verified' and failure_reason=None."""
    verbatim_text = "Metformin hydrochloride tablets are contraindicated in patients with: Severe renal impairment (eGFR below 30 mL/min/1.73 m 2 )"
    citation = Citation(
        source_name="Metformin FDA Label",
        section="contraindications",
        verbatim_quote=verbatim_text,
    )

    diagnostic = verify_citation_detailed(citation, [sample_metformin_chunk])

    assert diagnostic.verification_result == "verified"
    assert diagnostic.failure_reason is None
    assert diagnostic.similarity_score >= 0.95
    assert diagnostic.matched_chunk_id == "fda_label_Metformin_contraindications_0000"
    assert diagnostic.verification_method in ["exact_substring", "normalized_substring"]


# -----------------------------------------------------------------------------
# Test 2: Mismatched Quotations (Paraphrasing / Term Substitution)
# -----------------------------------------------------------------------------
def test_2_mismatched_quotation_wording(sample_metformin_chunk):
    """Assert paraphrased quotation (e.g. parametric memory substitution) is marked unverified with specific failure reason."""
    paraphrased_text = (
        "Metformin hydrochloride tablets are contraindicated in patients with an estimated "
        "glomerular filtration rate (eGFR) below 30 mL/minute/1.73 m 2"
    )
    citation = Citation(
        source_name="Metformin FDA Label",
        section="contraindications",
        verbatim_quote=paraphrased_text,
    )

    diagnostic = verify_citation_detailed(citation, [sample_metformin_chunk])

    assert diagnostic.verification_result == "unverified"
    assert diagnostic.failure_reason is not None
    assert "diverges from source text" in diagnostic.failure_reason or "not found" in diagnostic.failure_reason
    assert diagnostic.similarity_score < 0.88


# -----------------------------------------------------------------------------
# Test 3: Incorrect Source Metadata (Section or Source Mismatch)
# -----------------------------------------------------------------------------
def test_3_incorrect_source_metadata(sample_metformin_chunk):
    """Assert quotation found in section A but cited under section B generates metadata mismatch diagnostic."""
    verbatim_text = "Metformin hydrochloride tablets are contraindicated in patients with: Severe renal impairment (eGFR below 30 mL/min/1.73 m 2 )"
    
    # Intentionally cite under 'drug_interactions' instead of 'contraindications'
    citation = Citation(
        source_name="Metformin FDA Label",
        section="drug_interactions",
        verbatim_quote=verbatim_text,
    )

    diagnostic = verify_citation_detailed(citation, [sample_metformin_chunk])

    assert diagnostic.verification_result == "unverified"
    assert diagnostic.verification_method == "metadata_mismatch"
    assert diagnostic.failure_reason is not None
    assert "Source metadata mismatch" in diagnostic.failure_reason
    assert "contraindications" in diagnostic.failure_reason
    assert "drug_interactions" in diagnostic.failure_reason
    assert diagnostic.matched_chunk_id == "fda_label_Metformin_contraindications_0000"


# -----------------------------------------------------------------------------
# Test 4: Unsupported Claims with Valid Quotations (Decoupling Requirement 2)
# -----------------------------------------------------------------------------
def test_4_unsupported_claim_with_valid_quotation(sample_paxlovid_chunk):
    """Assert quotation matches verbatim (verified=True), but contradictory clinical claim is marked claim_support_status='unsupported'."""
    verbatim_quote = "In patients with moderate renal impairment (eGFR >= 30 to < 60 mL/min), the dosage of PAXLOVID is 150 mg nirmatrelvir and 100 mg ritonavir twice daily for 5 days."
    citation = Citation(
        source_name="Paxlovid FDA Label",
        section="dosage_and_administration",
        verbatim_quote=verbatim_quote,
    )

    # Contradictory clinical assertion: claims Paxlovid is completely safe with no dose reduction
    contradictory_clinical_claim = (
        "Paxlovid is completely safe in moderate renal impairment and no dose reduction is needed. "
        "The standard dose is permitted without any restriction."
    )

    diagnostic = verify_citation_detailed(
        citation=citation,
        context_chunks=[sample_paxlovid_chunk],
        clinical_assertions=contradictory_clinical_claim,
    )

    # Quotation verification and claim grounding MUST be separated:
    assert diagnostic.verification_result == "verified", "Quotation is verbatim accurate in source text"
    assert diagnostic.claim_support_status == "unsupported", "Clinical claim contradicts evidence and must be unsupported"
    assert diagnostic.claim_support_reason is not None
    assert "Contradiction" in diagnostic.claim_support_reason or "does not provide clinical evidence" in diagnostic.claim_support_reason


# -----------------------------------------------------------------------------
# Test 5: System-Diagnostic Question Routing & Root Cause Explanation
# -----------------------------------------------------------------------------
def test_5_system_diagnostic_query(agent_app):
    """Query: 'Why is the Metformin citation marked UNVERIFIED CITATION?'
    - Assert classified as is_diagnostic=True, is_clinical=False.
    - Assert Evidence Gate is NOT triggered to block (evidence_gate_passed=True).
    - Assert final_assessment has is_diagnostic_response=True and blocked_by_guardrail=False.
    - Assert explanation identifies the exact root cause: wording divergence and section mismatch.
    - Assert zero secrets or API keys are exposed.
    """
    query = "Why is the Metformin citation marked UNVERIFIED CITATION?"
    result = agent_app.invoke({"query": query})
    assessment: ClinicalAssessment = result.get("final_assessment")

    assert result.get("is_diagnostic") is True, "Query must be classified as is_diagnostic=True"
    assert result.get("is_clinical") is False, "Diagnostic query must NOT be classified as clinical query"
    assert result.get("evidence_gate_passed") is True, "Diagnostic query must pass Evidence Gate"

    assert assessment is not None, "Must produce a final ClinicalAssessment"
    assert assessment.is_diagnostic_response is True, "Assessment must flag is_diagnostic_response=True"
    assert assessment.blocked_by_guardrail is False, "Must not be blocked by guardrail"

    # Verify diagnostic explanation content
    explanation_text = assessment.clinical_mechanism + " " + assessment.dosing_and_actionable_guidance
    assert "Severe renal impairment" in explanation_text or "eGFR" in explanation_text
    assert "mL/min" in explanation_text or "paraphras" in explanation_text.lower()
    assert "Separation of Citation Verification" in explanation_text or "Claim Grounding" in explanation_text

    # Verify structured diagnostics attached
    assert len(assessment.verification_diagnostics) >= 1
    diag = assessment.verification_diagnostics[0]
    assert diag.verification_result == "unverified"
    assert diag.failure_reason is not None
    assert diag.quoted_text is not None

    # Strict secret and sensitive data hygiene
    for secret_pattern in [r"gsk_[A-Za-z0-9_\-]+", r"sk-[A-Za-z0-9_\-]+", r"Bearer\s+"]:
        assert not re.search(secret_pattern, explanation_text), "Leaked API key or secret in diagnostic response!"


# -----------------------------------------------------------------------------
# Test 6: Evidence Gate Preservation for Clinical Queries
# -----------------------------------------------------------------------------
def test_6_evidence_gate_preserved_for_clinical_queries(agent_app):
    """Assert clinical queries with unsupported combinations continue to trigger Evidence Gate and get blocked."""
    unsupported_query = "Can a patient take Paxlovid with Unobtainium-999?"
    result = agent_app.invoke({"query": unsupported_query})
    assessment: ClinicalAssessment = result.get("final_assessment")

    assert result.get("is_clinical") is True, "Clinical query must have is_clinical=True"
    assert result.get("is_diagnostic", False) is False, "Clinical query must NOT be diagnostic"
    assert assessment is not None
    assert assessment.blocked_by_guardrail is True or assessment.risk_severity == "Insufficient Evidence / Cannot Verify"
    assert assessment.groundedness_score == 0.0


# -----------------------------------------------------------------------------
# Test 7: Concept Stop-Words Filtering & Numbered List UI Formatting
# -----------------------------------------------------------------------------
def test_7_concept_stopwords_filtering_and_ui_formatting(sample_paxlovid_chunk):
    """Verify stop words are filtered from shared concepts and numbered guidance is cleanly formatted."""
    from src.agent.guardrails import is_clinical_concept_token, verify_citation_detailed
    from src.ui.app import format_actionable_guidance_html, sanitize_shared_concepts_display

    # 1. Verify Stop-Words Filtering
    forbidden_stop_words = [
        "with", "below", "above", "from", "into", "during", "until", "against", "among",
        "throughout", "despite", "towards", "upon", "concerning", "that", "this", "these",
        "those", "have", "should", "would", "could", "patients", "table", "shown", "taking",
        "used", "dose", "dosage", "the", "and", "min", "sec", "for", "are", "is",
    ]
    for word in forbidden_stop_words:
        assert not is_clinical_concept_token(word), f"Stop word '{word}' must not be treated as a clinical concept!"

    clinical_terms = [
        "impairment", "egfr", "moderate", "renal", "paxlovid", "metformin", "contraindicated", "reduction",
    ]
    for term in clinical_terms:
        assert is_clinical_concept_token(term), f"Clinical term '{term}' should be recognized as a valid concept!"

    # Short permitted acronyms
    assert is_clinical_concept_token("cr") is True
    assert is_clinical_concept_token("gfr") is True

    # 2. Verify shared concepts extraction in CitationVerificationDiagnostic
    citation = Citation(
        source_name="Paxlovid FDA Label",
        section="dosage_and_administration",
        verbatim_quote="In patients with moderate renal impairment (eGFR >= 30 to < 60 mL/min), the dosage of PAXLOVID is 150 mg nirmatrelvir and 100 mg ritonavir twice daily for 5 days.",
    )
    assertions = (
        "In patients with moderate renal impairment, Paxlovid requires a 50% dose reduction. "
        "Prescribe 150 mg nirmatrelvir with 100 mg ritonavir twice daily."
    )
    diagnostic = verify_citation_detailed(
        citation=citation,
        context_chunks=[sample_paxlovid_chunk],
        clinical_assertions=assertions,
    )
    assert diagnostic.claim_support_status == "supported"
    assert diagnostic.claim_support_reason is not None

    # Crucial assertion: shared concepts must NOT contain stop words
    for forbidden in ["with", "below", "patients", "the", "dosage", "dose", "min"]:
        assert f" {forbidden}," not in diagnostic.claim_support_reason
        assert f": {forbidden}," not in diagnostic.claim_support_reason
        assert f" {forbidden})" not in diagnostic.claim_support_reason

    # Must contain clinical terms
    assert any(term in diagnostic.claim_support_reason for term in ["moderate", "renal", "impairment", "paxlovid"])

    # UI sanitizer check
    raw_diagnostic_text = "Quotation directly substantiates clinical assertions (shared concepts: with, below, impairment, egfr, moderate)."
    sanitized = sanitize_shared_concepts_display(raw_diagnostic_text)
    assert "with" not in sanitized
    assert "below" not in sanitized
    assert "impairment" in sanitized
    assert "moderate" in sanitized

    # 3. Verify Numbered List UI HTML Formatting
    run_on_guidance = (
        "1. In moderate renal impairment, Paxlovid requires dose reduction: 150 mg nirmatrelvir twice daily. "
        "2. Metformin may be maintained if baseline eGFR is >= 45 mL/min. "
        "3. Monitor renal function and signs of lactic acidosis."
    )
    html_output = format_actionable_guidance_html(run_on_guidance)
    assert '<span style="font-weight: 700; color: #0f766e; min-width: 1.85rem; display: inline-block;">1.</span>' in html_output
    assert '<span style="font-weight: 700; color: #0f766e; min-width: 1.85rem; display: inline-block;">2.</span>' in html_output
    assert '<span style="font-weight: 700; color: #0f766e; min-width: 1.85rem; display: inline-block;">3.</span>' in html_output
    assert 'display: flex; align-items: flex-start;' in html_output

