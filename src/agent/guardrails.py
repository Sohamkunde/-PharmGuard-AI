"""Clinical Guardrail Verifier for PharmGuard.

Implements strict claim-level groundedness, verbatim citation verification,
and the post-generation evidence gate.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

try:
    import rapidfuzz
    from rapidfuzz import fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

from src.agent.schemas import ClinicalAssessment, Citation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("guardrails")


def normalize_clinical_text(text: str) -> str:
    """Normalize text for verbatim quote and claim matching."""
    if not text:
        return ""
    # Normalize unicode spaces and hyphens
    t = (
        text.lower()
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    # Strip excessive punctuation and collapse whitespace
    t = re.sub(r"[^\w\s\-\.]", " ", t)
    return " ".join(t.split())


from src.agent.schemas import ClinicalAssessment, Citation, CitationVerificationDiagnostic


def _is_section_compatible(cited_sec: str, cited_src: str, chunk_sec: str) -> bool:
    """Check if the cited section and chunk section are mutually compatible."""
    chunk_sec_clean = chunk_sec.lower().replace("_", " ").strip()
    cited_sec_clean = cited_sec.lower().replace("_", " ").strip()
    cited_src_clean = cited_src.lower().replace("_", " ").strip()

    # 1. Exact or substring match
    if chunk_sec_clean in cited_sec_clean or cited_sec_clean in chunk_sec_clean:
        return True
    if chunk_sec_clean in cited_src_clean:
        return True
    if chunk_sec.lower() in cited_src.lower():
        return True

    # 2. Key FDA section keywords
    chunk_words = {w for w in re.findall(r"\b[a-z]{3,}\b", chunk_sec_clean) if w not in {"and", "for", "the", "use", "with"}}
    cited_words = {w for w in re.findall(r"\b[a-z]{3,}\b", cited_sec_clean) if w not in {"and", "for", "the", "use", "with"}}
    if chunk_words and cited_words:
        if chunk_words.issubset(cited_words) or len(chunk_words.intersection(cited_words)) >= 1:
            return True

    # 3. Known FDA subsection alignments
    if "renal" in cited_sec_clean and ("renal" in chunk_sec_clean or "dosage" in chunk_sec_clean or "warning" in chunk_sec_clean or "contraindication" in chunk_sec_clean):
        return True
    if "alert" in cited_sec_clean and "calculator" in chunk_sec_clean:
        return True
    if "contraindication" in chunk_sec_clean and "contraindication" in cited_sec_clean:
        return True
    if "warning" in chunk_sec_clean and ("warning" in cited_sec_clean or "caution" in cited_sec_clean):
        return True
    if "interaction" in chunk_sec_clean and "interaction" in cited_sec_clean:
        return True

    return False


def _is_source_compatible(cited_src: str, cited_sec: str, chunk_drug: str, chunk_source_type: str) -> bool:
    """Check if the cited source and chunk drug/source are mutually compatible."""
    chunk_drug_clean = chunk_drug.lower().strip()
    cited_src_clean = cited_src.lower().strip()
    cited_sec_clean = cited_sec.lower().strip()

    if not chunk_drug_clean:
        return True
    if chunk_drug_clean in cited_src_clean or cited_src_clean in chunk_drug_clean:
        return True
    if chunk_drug_clean in cited_sec_clean:
        return True
    if "calculator" in chunk_drug_clean and ("calculator" in cited_src_clean or "egfr" in cited_src_clean or "alert" in cited_sec_clean):
        return True
    if "guideline" in chunk_drug_clean or chunk_source_type == "clinical_guideline":
        if "guideline" in cited_src_clean or "nih" in cited_src_clean or "kdigo" in cited_src_clean or "fda" in cited_src_clean:
            return True

    # Check drug aliases (e.g. nirmatrelvir / ritonavir / paxlovid)
    try:
        from src.agent.entity_resolver import get_entity_resolver
        resolver = get_entity_resolver()
        aliases = resolver.expand_drugs([chunk_drug_clean]).get(chunk_drug_clean, [])
        if any(a in cited_src_clean for a in aliases):
            return True
    except Exception:
        pass

    return False


# Comprehensive stop words that should NEVER appear in clinical 'shared concepts'
CONCEPT_STOPWORDS = {
    # Prepositions, Conjunctions & Articles
    "with", "below", "above", "from", "into", "during", "until", "against", "among",
    "throughout", "despite", "towards", "upon", "concerning", "about", "after", "before",
    "between", "through", "under", "over", "within", "without", "along", "across",
    "behind", "beyond", "around", "near", "since", "than", "whether", "while", "where",
    "when", "whereas", "although", "because", "unless", "that", "this", "these", "those",
    "the", "and", "for", "are", "is", "was", "were", "been", "being", "has", "had",
    "have", "having", "does", "did", "doing", "done", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "as", "at", "by", "down", "except", "front", "further", "here",
    "how", "in", "inside", "near", "next", "of", "off", "on", "onto", "out", "outside",
    "past", "till", "to", "toward", "underneath", "up", "then", "there", "therefore",
    "thus", "what", "whatever", "whence", "whenever", "whereafter", "whereby", "wherein",
    "whereupon", "wherever", "which", "whichever", "whither", "who", "whoever", "whole",
    "whom", "whose", "why", "either", "neither", "but",
    # Pronouns
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "we", "us", "our", "ours",
    "ourselves", "they", "them", "their", "theirs", "themselves", "i", "me", "my", "myself",
    "anyone", "someone", "everyone", "nobody", "anything", "something", "everything",
    # Auxiliary & Common Non-clinical Verbs
    "taking", "take", "takes", "taken", "used", "uses", "using", "make", "makes", "making",
    "made", "show", "shown", "shows", "showing", "give", "given", "giving", "need", "needs",
    "needed", "include", "included", "includes", "including", "contain", "contains",
    "contained", "containing", "indicate", "indicated", "indicates", "indicating",
    "regard", "regarded", "observe", "observed", "observes", "report", "reported",
    "reports", "reporting", "allow", "allows", "allowed", "start", "starts", "started",
    "stop", "stops", "stopped", "appear", "appears", "appeared", "ensure", "ensures",
    "consider", "considered", "considering", "provide", "provides", "provided", "providing",
    "require", "requires", "required", "requiring", "assess", "assessed", "assessing",
    # Units, Time Intervals, & Quantifiers (Not substantive clinical concepts)
    "min", "sec", "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days",
    "daily", "week", "weeks", "month", "months", "year", "years", "time", "times", "per",
    "rate", "total", "mg", "mcg", "ml", "kg", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten", "first", "second", "third",
    # Generic Clinical/Structural Fillers (Non-pharmacological / Non-anatomical text words)
    "patients", "patient", "table", "tables", "figure", "figures", "section", "sections",
    "label", "labels", "insert", "package", "prescribing", "information", "study", "studies",
    "trial", "trials", "data", "results", "result", "group", "groups", "subject", "subjects",
    "dose", "doses", "dosage", "dosing", "once", "twice", "administer", "administered",
    "administration", "orally", "oral", "tablet", "tablets", "capsule", "capsules",
    "based", "recommended", "recommendation", "recommendations", "prescribed", "prescribe",
    "follow", "following", "followed", "guidance", "guidelines", "protocol", "protocols",
    "level", "levels", "value", "values", "case", "cases", "type", "types", "form", "forms",
    "line", "lines", "standard", "note", "notes", "marked", "unmarked", "known", "unknown",
    "treatment", "therapy", "regimen", "drug", "drugs", "medication", "medications",
    "medicine", "medicines", "product", "products", "article",
}

# Permitted clinical abbreviations and short tokens (<= 2 chars)
CLINICAL_SHORT_TOKENS = {"mg", "iv", "po", "sc", "im", "pr", "cr", "gfr"}


def is_clinical_concept_token(token: str) -> bool:
    """Determine whether a token is a substantive clinical, pharmacological, or anatomical concept."""
    t = token.lower().strip(".-,:;()[]{}'\"")
    if not t:
        return False
    if t in CONCEPT_STOPWORDS:
        return False
    if len(t) < 3 and t not in CLINICAL_SHORT_TOKENS:
        return False
    if t.isdigit():
        return False
    # Filter out pure numbers or metric units like '30', '60mg', '150mg', '30ml'
    if re.match(r"^\d+(?:\.\d+)?(?:mg|g|mcg|ml|kg|l|min)?$", t):
        return False
    return True


def verify_citation_detailed(
    citation: Citation,
    context_chunks: List[Dict[str, Any]],
    clinical_assertions: Optional[str] = None,
) -> CitationVerificationDiagnostic:
    """Perform granular verbatim verification and claim-support check for a citation.

    Identifies exact failure causes:
    - Empty or whitespace quote
    - Quotation wording mismatch (paraphrasing, modified terms, ratio < 88%)
    - Source metadata mismatch (quote exists in source A/section X, but cited as B/Y)
    - Not found in any context window
    - Separates citation-verification status from clinical-answer grounding status
    """
    quote = citation.verbatim_quote
    if not quote or not quote.strip():
        return CitationVerificationDiagnostic(
            source_name=citation.source_name,
            section=citation.section,
            quoted_text=quote or "",
            verification_method="none",
            verification_result="unverified",
            failure_reason="Quotation text is empty or whitespace.",
            similarity_score=0.0,
            claim_support_status="unsupported",
            claim_support_reason="Cannot substantiate clinical claim with an empty quotation.",
        )

    if not context_chunks:
        return CitationVerificationDiagnostic(
            source_name=citation.source_name,
            section=citation.section,
            quoted_text=quote,
            verification_method="none",
            verification_result="unverified",
            failure_reason="No retrieved context chunks available for verification.",
            similarity_score=0.0,
            claim_support_status="unsupported",
            claim_support_reason="No evidence context provided to support claim.",
        )

    norm_quote = normalize_clinical_text(quote)
    if not norm_quote:
        return CitationVerificationDiagnostic(
            source_name=citation.source_name,
            section=citation.section,
            quoted_text=quote,
            verification_method="none",
            verification_result="unverified",
            failure_reason="Quotation contains only punctuation or non-alphanumeric characters.",
            similarity_score=0.0,
            claim_support_status="unsupported",
            claim_support_reason="Quotation lacks verifiable textual content.",
        )

    best_ratio = 0.0
    best_chunk_id: Optional[str] = None
    best_chunk_source = ""
    best_chunk_section = ""
    best_snippet = ""

    matched_in_cited_section: Optional[Tuple[str, str, float]] = None
    matched_in_other_chunk: Optional[Tuple[str, str, str, str, float]] = None

    for chunk in context_chunks:
        chunk_id = chunk.get("chunk_id", "unknown")
        meta = chunk.get("metadata", {})
        chunk_drug = str(meta.get("drug_name", "")).strip().lower()
        chunk_sec = str(meta.get("section", "")).strip().lower()

        window_raw = (
            meta.get("window")
            or chunk.get("window")
            or chunk.get("sentence", "")
        )
        if not window_raw:
            continue

        norm_window = normalize_clinical_text(window_raw)

        match_method: Optional[str] = None
        match_score = 0.0

        # 1. Exact raw substring
        if quote.strip().lower() in window_raw.lower():
            match_method = "exact_substring"
            match_score = 1.0
        # 2. Normalized substring
        elif norm_quote in norm_window:
            match_method = "normalized_substring"
            match_score = 1.0
        # 3. Discontinuous ellipsis match
        elif "..." in quote or "…" in quote:
            parts = [
                normalize_clinical_text(p)
                for p in re.split(r"\.\.\.|\u2026", quote)
                if len(normalize_clinical_text(p)) >= 6
            ]
            if parts and all(
                (p in norm_window or (_RAPIDFUZZ_AVAILABLE and fuzz.partial_ratio(p, norm_window) >= 85.0))
                for p in parts
            ):
                match_method = "ellipsis_segment_match"
                match_score = 0.98
        # 4. Fuzzy ratio
        elif _RAPIDFUZZ_AVAILABLE:
            ratio = fuzz.partial_ratio(norm_quote, norm_window)
            token_ratio = fuzz.token_set_ratio(norm_quote, norm_window)
            if ratio >= 88.0:
                match_method = "fuzzy_partial_ratio"
                match_score = round(ratio / 100.0, 4)
            elif token_ratio >= 92.0 and ratio >= 60.0:
                match_method = "fuzzy_token_set_ratio"
                match_score = round(token_ratio / 100.0, 4)

            if ratio > best_ratio:
                best_ratio = ratio
                best_chunk_id = chunk_id
                best_chunk_source = meta.get("drug_name", "")
                best_chunk_section = meta.get("section", "")
                best_snippet = window_raw[:100]
        else:
            quote_words = set(norm_quote.split())
            if len(quote_words) >= 4:
                window_words = set(norm_window.split())
                matched_w = len(quote_words.intersection(window_words))
                r = matched_w / len(quote_words)
                if r >= 0.85:
                    match_method = "token_overlap"
                    match_score = round(r, 4)
                if (r * 100.0) > best_ratio:
                    best_ratio = r * 100.0
                    best_chunk_id = chunk_id
                    best_chunk_source = meta.get("drug_name", "")
                    best_chunk_section = meta.get("section", "")
                    best_snippet = window_raw[:100]

        if match_method:
            src_type = str(meta.get("source_type", "")).strip().lower()
            sec_matches = _is_section_compatible(citation.section, citation.source_name, chunk_sec)
            src_matches = _is_source_compatible(citation.source_name, citation.section, chunk_drug, src_type)

            if sec_matches and src_matches:
                matched_in_cited_section = (chunk_id, match_method, match_score)
                break
            elif not matched_in_other_chunk:
                matched_in_other_chunk = (
                    chunk_id,
                    meta.get("drug_name", "Unknown"),
                    meta.get("section", "Unknown"),
                    match_method,
                    match_score,
                )

    if matched_in_cited_section:
        cid, method, score = matched_in_cited_section
        res_status = "verified"
        fail_reason = None
        matched_id = cid
        ver_method = method
        sim_score = score
    elif matched_in_other_chunk:
        cid, o_drug, o_sec, method, score = matched_in_other_chunk
        res_status = "unverified"
        fail_reason = (
            f"Source metadata mismatch: quotation exists in section '{o_sec}' of '{o_drug}' (chunk {cid}), "
            f"but was cited under section '{citation.section}' of '{citation.source_name}'."
        )
        matched_id = cid
        ver_method = "metadata_mismatch"
        sim_score = score
    else:
        res_status = "unverified"
        ver_method = "unmatched"
        matched_id = None
        sim_score = round(best_ratio / 100.0, 4)
        if best_ratio >= 40.0:
            fail_reason = (
                f"Quotation wording diverges from source text (similarity: {best_ratio:.1f}% < 88.0%). "
                f"Words in quote do not match retrieved text in '{best_chunk_id}'."
            )
        else:
            fail_reason = (
                f"Quotation text was not found in any retrieved context window for '{citation.source_name}'."
            )

    # Evaluate Claim Support (Requirement 2)
    claim_status = "not_evaluated"
    claim_reason = None
    if res_status == "unverified":
        claim_status = "unsupported"
        claim_reason = f"Cannot ground claim: citation is unverified ({fail_reason})."
    elif clinical_assertions:
        norm_assertions = normalize_clinical_text(clinical_assertions)

        # Drug-specific sentence filtering to prevent cross-drug false contradictions
        relevant_assertions = norm_assertions
        if citation.source_name and citation.source_name != "Unknown":
            clean_src = re.sub(r"[^A-Za-z0-9\s]", " ", citation.source_name).strip()
            src_first = clean_src.split()[0].lower() if clean_src else ""
            if src_first and src_first not in {"clinical", "fda", "nih", "cdc", "unknown"}:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", norm_assertions) if s.strip()]
                drug_sentences = [
                    s for s in sentences
                    if src_first in s or "both" in s or "combination" in s or "concomitant" in s or "coadministration" in s
                ]
                if drug_sentences:
                    relevant_assertions = " ".join(drug_sentences)

        contra_words = {"contraindicated", "fatal", "avoid", "toxicity", "discontinue", "severe risk", "strict washout"}
        safe_words = {"safe", "no interaction", "no conflict", "no clinically significant", "permitted", "compatible", "well tolerated"}

        quote_has_contra = bool(any(w in norm_quote for w in contra_words))
        claim_has_safe = bool(any(w in relevant_assertions for w in safe_words))
        quote_has_safe = bool(any(w in norm_quote for w in safe_words))
        claim_has_contra = bool(any(w in relevant_assertions for w in contra_words))

        quote_has_dose_mod = bool(any(w in norm_quote for w in ["reduce", "reduction", "150 mg", "dosage", "adjustment", "adjust"]))
        claim_claims_no_reduction = bool(any(w in relevant_assertions for w in [
            "no dose reduction", "without dose reduction", "no reduction", "without restriction", "no restriction",
            "standard dose is permitted", "no adjustment", "without adjustment"
        ]))

        if quote_has_contra and claim_has_safe:
            claim_status = "unsupported"
            claim_reason = "Contradiction: quotation explicitly mandates contraindication or severe toxicity, but clinical claim asserts combination is safe."
        elif quote_has_safe and claim_has_contra:
            claim_status = "unsupported"
            claim_reason = "Contradiction: quotation indicates no clinically significant interaction, but claim asserts severe contraindication."
        elif quote_has_dose_mod and claim_claims_no_reduction:
            claim_status = "unsupported"
            claim_reason = "Contradiction: quotation mandates specific renal dose adjustment, but clinical claim asserts no dose reduction is needed."
        else:
            q_tokens = [w.lower().strip("-") for w in re.findall(r"\b[A-Za-z0-9\-]+\b", norm_quote)]
            q_concepts = [w for w in q_tokens if is_clinical_concept_token(w)]
            claim_tokens = set(w.lower().strip("-") for w in re.findall(r"\b[A-Za-z0-9\-]+\b", relevant_assertions) if is_clinical_concept_token(w))

            # Deduplicate substantive concepts while preserving order of occurrence
            seen_concepts = set()
            shared_concepts = []
            for w in q_concepts:
                if w in claim_tokens and w not in seen_concepts:
                    seen_concepts.add(w)
                    shared_concepts.append(w)

            if len(shared_concepts) >= 2:
                claim_status = "supported"
                claim_reason = f"Quotation directly substantiates clinical assertions (shared concepts: {', '.join(shared_concepts[:4])})."
            else:
                claim_status = "unsupported"
                claim_reason = "Quotation is verified verbatim in source, but does not provide clinical evidence substantiating the specific assertions in this recommendation."

    return CitationVerificationDiagnostic(
        source_name=citation.source_name,
        section=citation.section,
        quoted_text=quote,
        matched_chunk_id=matched_id,
        verification_method=ver_method,
        verification_result=res_status,
        failure_reason=fail_reason,
        similarity_score=sim_score,
        claim_support_status=claim_status,
        claim_support_reason=claim_reason,
    )


def verify_citation(quote: str, context_chunks: List[Dict[str, Any]]) -> bool:
    """Verify that a verbatim quote exists in retrieved context windows (backward-compatible)."""
    temp_cit = Citation(source_name="Unknown", section="Unknown", verbatim_quote=quote)
    diag = verify_citation_detailed(temp_cit, context_chunks)
    return diag.verification_result == "verified"


def verify_claim_entailment(
    text: str,
    context_chunks: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Check sentence-level clinical token entailment against retrieved context.

    Extracts numbers/dosages (e.g. '150 mg'), clinical parameters (e.g. 'eGFR', 'CYP3A4'),
    and verifies their presence in retrieved context windows.
    """
    if not text or not context_chunks:
        return True, []

    combined_context = " ".join([
        (c.get("metadata", {}).get("window") or c.get("window") or c.get("sentence", ""))
        for c in context_chunks
    ]).lower()
    norm_context = normalize_clinical_text(combined_context)

    # Split into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    unverified_claims = []

    dosage_pattern = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|days|weeks|hours)\b", re.IGNORECASE)
    clinical_markers = [
        "egfr", "crcl", "creatinine", "cyp3a4", "cyp2d6", "cyp", "p-gp",
        "lactic acidosis", "serotonin syndrome", "rhabdomyolysis", "contraindicated",
    ]

    for sentence in sentences:
        norm_s = normalize_clinical_text(sentence)
        # Check dosages
        dosages = dosage_pattern.findall(sentence)
        for dos in dosages:
            norm_dos = normalize_clinical_text(dos)
            if norm_dos and norm_dos not in norm_context:
                unverified_claims.append(f"Unverified dosage '{dos}' in: '{sentence[:60]}...'")

        # Check key biomarkers
        for marker in clinical_markers:
            if marker in norm_s and marker not in norm_context:
                unverified_claims.append(f"Unverified clinical marker '{marker}' in: '{sentence[:60]}...'")

    passed = len(unverified_claims) == 0
    return passed, unverified_claims


def guardrail_verifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Verify citations, claim-level entailment, and enforce post-generation gate.

    If groundedness_score < 0.75 or len(verified_citations) == 0,
    the generated recommendation is immediately overridden and blocked.
    """
    assessment: Optional[ClinicalAssessment] = state.get("final_assessment")
    chunks = state.get("retrieved_chunks", [])

    if assessment is None:
        logger.error("Guardrail Verifier: No final_assessment found in state.")
        return {"error": "Missing clinical assessment in guardrail verifier."}

    # Pass-through system-diagnostic queries (explaining verification trace, not medical advice)
    if state.get("is_diagnostic", False):
        assessment.blocked_by_guardrail = False
        assessment.is_diagnostic_response = True
        return {"final_assessment": assessment}

    # Pass-through non-clinical conversational / out-of-domain assessments
    if not state.get("is_clinical", True):
        assessment.blocked_by_guardrail = False
        return {"final_assessment": assessment}

    # If already marked as Insufficient Evidence by the Pre-Generation Gate
    if assessment.risk_severity == "Insufficient Evidence / Cannot Verify":
        assessment.blocked_by_guardrail = True
        assessment.groundedness_score = 0.0
        assessment.citation_verification_score = 0.0
        assessment.claim_grounding_score = 0.0
        return {"final_assessment": assessment}

    combined_clinical_assertions = (
        f"{assessment.clinical_mechanism} {assessment.dosing_and_actionable_guidance}"
    )

    # 1. Verbatim Citation Verification & Claim Grounding Check
    verified_citations = []
    supported_citations = []
    diagnostics: List[CitationVerificationDiagnostic] = []

    if assessment.citations:
        for cit in assessment.citations:
            diag = verify_citation_detailed(cit, chunks, combined_clinical_assertions)
            cit.verified = (diag.verification_result == "verified")
            cit.verification_method = diag.verification_method
            cit.matched_chunk_id = diag.matched_chunk_id
            cit.failure_reason = diag.failure_reason
            cit.claim_support_status = diag.claim_support_status
            cit.diagnostic = diag
            diagnostics.append(diag)

            if cit.verified:
                verified_citations.append(cit)
            else:
                logger.warning(
                    "Guardrail Verifier: Citation failed verification [%s (%s)]: '%s' (Reason: %s)",
                    cit.source_name,
                    cit.section,
                    cit.verbatim_quote[:60],
                    diag.failure_reason,
                )

            if diag.claim_support_status == "supported":
                supported_citations.append(cit)

        total_citations = len(assessment.citations)
        c_ver_score = len(verified_citations) / total_citations if total_citations > 0 else 0.0
        c_grd_score = len(supported_citations) / total_citations if total_citations > 0 else 0.0

        assessment.citation_verification_score = round(c_ver_score, 4)
        assessment.claim_grounding_score = round(c_grd_score, 4)
        assessment.groundedness_score = round(c_ver_score, 4)
        assessment.verification_diagnostics = diagnostics
    else:
        assessment.groundedness_score = 0.0
        assessment.citation_verification_score = 0.0
        assessment.claim_grounding_score = 0.0
        assessment.verification_diagnostics = []

    # 2. Claim-Level Entailment Check
    entailment_passed_mech, unverified_mech = verify_claim_entailment(
        assessment.clinical_mechanism,
        chunks,
    )
    entailment_passed_dose, unverified_dose = verify_claim_entailment(
        assessment.dosing_and_actionable_guidance,
        chunks,
    )
    if unverified_mech or unverified_dose:
        logger.warning(
            "Guardrail Verifier: Claim-level entailment issues detected: %s",
            unverified_mech + unverified_dose,
        )

    # 3. Hard Post-Generation Gate
    # Enforce: groundedness_score >= 0.75 AND len(verified_citations) > 0
    gate_failed = (
        assessment.groundedness_score < 0.75
        or len(verified_citations) == 0
    )

    if gate_failed:
        logger.error(
            "Guardrail Verifier GATE TRIGGERED: Groundedness=%.1f%%, Verified Citations=%d. Overriding output.",
            assessment.groundedness_score * 100,
            len(verified_citations),
        )
        assessment.risk_severity = "Insufficient Evidence / Cannot Verify"
        assessment.clinical_mechanism = (
            "Output suppressed by PharmGuard Groundedness Gate: Generated claims could not be verified "
            "against retrieved FDA/guideline text."
        )
        assessment.dosing_and_actionable_guidance = (
            "Action Blocked: Do not rely on automated output. Consult official prescribing information "
            "or a clinical pharmacist."
        )
        assessment.blocked_by_guardrail = True
    else:
        assessment.blocked_by_guardrail = False
        logger.info(
            "Guardrail Verifier: PASSED (Groundedness=%.1f%%, Verified Citations=%d/%d, Supported Claims=%d/%d)",
            assessment.groundedness_score * 100,
            len(verified_citations),
            len(assessment.citations),
            len(supported_citations),
            len(assessment.citations),
        )

    return {
        "final_assessment": assessment,
        "verification_diagnostics": [d.model_dump() for d in diagnostics],
    }
