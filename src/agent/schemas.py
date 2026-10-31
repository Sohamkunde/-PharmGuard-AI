"""Pydantic v2 schemas for PharmGuard clinical assessment and groundedness verification."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class CitationVerificationDiagnostic(BaseModel):
    """Detailed verification diagnostic for a single citation."""

    source_name: str = Field(..., description="Target source name (e.g. 'Metformin FDA Label').")
    section: str = Field(..., description="Target section (e.g. 'contraindications').")
    quoted_text: str = Field(..., description="Quotation text evaluated for verification.")
    matched_chunk_id: Optional[str] = Field(default=None, description="ID of chunk where quotation was located, if found.")
    verification_method: str = Field(default="none", description="Method: exact_substring, normalized_substring, ellipsis_segment_match, fuzzy_partial_ratio, unmatched.")
    verification_result: Literal["verified", "unverified"] = Field(default="unverified", description="Whether quote passed verbatim source verification.")
    failure_reason: Optional[str] = Field(default=None, description="Precise cause of verification failure.")
    similarity_score: float = Field(default=0.0, description="Highest similarity score against candidate chunks (0.0 to 1.0).")
    claim_support_status: Literal["supported", "unsupported", "not_evaluated"] = Field(
        default="not_evaluated",
        description="Whether the quotation supports the associated clinical assertion in the assessment.",
    )
    claim_support_reason: Optional[str] = Field(default=None, description="Explanation of claim entailment or contradiction.")


class Citation(BaseModel):
    """Verbatim clinical citation linked to evidence source."""

    source_name: str = Field(
        ...,
        description="Name of the drug label or clinical practice guideline source (e.g. 'Paxlovid FDA Label', 'KDIGO Guideline').",
    )
    section: str = Field(
        ...,
        description="Specific clinical section (e.g. 'dosage_and_administration', 'contraindications', 'page_1').",
    )
    verbatim_quote: str = Field(
        ...,
        description="Exact verbatim excerpt from the evidence window supporting the clinical assertion.",
    )
    verified: bool = Field(
        default=False,
        description="Whether the verbatim quote was verified against the retrieved context by the guardrail verifier.",
    )
    verification_method: Optional[str] = Field(
        default=None,
        description="Method used to verify the quote (e.g. exact_substring, normalized_substring, fuzzy_partial_ratio).",
    )
    matched_chunk_id: Optional[str] = Field(
        default=None,
        description="ID of the retrieved chunk where the quotation matched.",
    )
    failure_reason: Optional[str] = Field(
        default=None,
        description="Specific cause if citation failed verification.",
    )
    claim_support_status: Literal["supported", "unsupported", "not_evaluated"] = Field(
        default="not_evaluated",
        description="Separated status indicating if verified quote supports the clinical claim.",
    )
    diagnostic: Optional[CitationVerificationDiagnostic] = Field(
        default=None,
        description="Full structured diagnostic record for this citation.",
    )


RiskSeverity = Literal[
    "Critical / Contraindicated",
    "Major / Dose Adjustment Required",
    "Moderate / Monitor",
    "Low / No Known Conflict",
    "Insufficient Evidence / Cannot Verify",
]


class ClinicalAssessment(BaseModel):
    """Structured clinical decision support assessment with groundedness validation."""

    risk_severity: RiskSeverity = Field(
        ...,
        description="Stratified risk severity level for the patient regimen.",
    )
    clinical_mechanism: str = Field(
        ...,
        description="Pharmacological, pharmacokinetic (CYP450/P-gp), or pharmacodynamic mechanism of conflict/toxicity.",
    )
    dosing_and_actionable_guidance: str = Field(
        ...,
        description="Specific dosage modifications, washout intervals, monitoring labs, or safe therapeutic alternatives.",
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of verbatim citations extracted directly from retrieved FDA labels or clinical guidelines.",
    )
    groundedness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall groundedness score (0.0 to 1.0).",
    )
    citation_verification_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of verbatim-verified citations to total citations (0.0 to 1.0).",
    )
    claim_grounding_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Entailment ratio of generated clinical claims supported by verified evidence (0.0 to 1.0).",
    )
    verification_diagnostics: List[CitationVerificationDiagnostic] = Field(
        default_factory=list,
        description="Structured verification diagnostics for all citations.",
    )
    blocked_by_guardrail: bool = Field(
        default=False,
        description="Whether output was suppressed/blocked due to failing evidence gates or groundedness checks.",
    )
    is_diagnostic_response: bool = Field(
        default=False,
        description="Whether this assessment is an explanatory system diagnostic response rather than clinical advice.",
    )

    def to_markdown_summary(self) -> str:
        """Render clinical assessment into clean Markdown report."""
        severity_badges = {
            "Critical / Contraindicated": "[CRITICAL / CONTRAINDICATED]",
            "Major / Dose Adjustment Required": "[MAJOR / DOSE ADJUSTMENT REQUIRED]",
            "Moderate / Monitor": "[MODERATE / MONITOR]",
            "Low / No Known Conflict": "[LOW / NO KNOWN CONFLICT]",
            "Insufficient Evidence / Cannot Verify": "[INSUFFICIENT EVIDENCE / CANNOT VERIFY]",
        }
        badge = severity_badges.get(self.risk_severity, f"[{self.risk_severity}]")
        if self.blocked_by_guardrail:
            badge += " (RECOMMENDATION BLOCKED)"
        
        md_lines = [
            f"### Clinical Risk Assessment: {badge}",
            f"**Groundedness Score:** {self.groundedness_score * 100:.1f}%\n",
            f"**Evidence Gate Status:** {'BLOCKED' if self.blocked_by_guardrail else 'PASSED'}\n",
            "#### 1. Mechanism of Action & Clinical Pharmacokinetics",
            self.clinical_mechanism,
            "",
            "#### 2. Actionable Dosing & Prescribing Guidance",
            self.dosing_and_actionable_guidance,
            "",
            "#### 3. Grounded Evidence Citations",
        ]
        
        if not self.citations:
            md_lines.append("_No citations attached._")
        else:
            for idx, c in enumerate(self.citations, 1):
                if c.verified:
                    status = "Verified"
                elif c.failure_reason:
                    status = f"Unverified: {c.failure_reason}"
                else:
                    status = "Unverified"
                md_lines.append(
                    f"{idx}. **{c.source_name}** ({c.section}) - *[{status}]*\n"
                    f"   > \"{c.verbatim_quote}\""
                )
        
        return "\n".join(md_lines)
