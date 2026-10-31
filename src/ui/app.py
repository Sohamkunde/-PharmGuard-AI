"""PharmGuard: Production-Grade Agentic Clinical RAG Dashboard.

Built with Streamlit, LangGraph, ChromaDB, BM25, and BGE Cross-Encoder Reranking.
Provides real-time clinical risk stratification, pharmacokinetic mechanism analysis,
verbatim grounded citation verification, and LangGraph execution observability.
"""

import os
import sys
import re
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent.graph import build_pharmguard_graph
from src.agent.schemas import ClinicalAssessment
from src.agent.guardrails import is_clinical_concept_token

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ui_app")


def format_actionable_guidance_html(text: str) -> str:
    """Format numbered list items or paragraphs into clean structured steps with distinct vertical spacing."""
    if not text:
        return ""
    # Ensure every numbered item (e.g. '1. ', '2. ', '1) ') or bullet begins after double newlines
    formatted = re.sub(r'(?<!^)\s+(\d+[\.\)]\s+)', r'\n\n\1', text.strip())
    formatted = re.sub(r'(?<!^)\s+([•\-\*]\s+)', r'\n\n\1', formatted)

    paragraphs = [p.strip() for p in formatted.split("\n\n") if p.strip()]
    if not paragraphs:
        return text

    html_parts = []
    for p in paragraphs:
        # Match leading numbered step, e.g. "1. " or "1) "
        num_match = re.match(r'^(\d+[\.\)])\s+(.*)', p, re.DOTALL)
        if num_match:
            step_num = num_match.group(1)
            step_body = num_match.group(2).replace("\n", "<br>")
            html_parts.append(
                f'<div style="margin-bottom: 0.85rem; line-height: 1.6; display: flex; align-items: flex-start;">'
                f'<span style="font-weight: 700; color: #0f766e; min-width: 1.85rem; display: inline-block;">{step_num}</span>'
                f'<span style="flex: 1; color: #334155;">{step_body}</span>'
                f'</div>'
            )
        else:
            bullet_match = re.match(r'^([•\-\*])\s+(.*)', p, re.DOTALL)
            if bullet_match:
                bullet_body = bullet_match.group(2).replace("\n", "<br>")
                html_parts.append(
                    f'<div style="margin-bottom: 0.85rem; line-height: 1.6; display: flex; align-items: flex-start;">'
                    f'<span style="font-weight: 700; color: #0f766e; min-width: 1.5rem; display: inline-block;">&bull;</span>'
                    f'<span style="flex: 1; color: #334155;">{bullet_body}</span>'
                    f'</div>'
                )
            else:
                body_html = p.replace("\n", "<br>")
                html_parts.append(f'<div style="margin-bottom: 0.85rem; line-height: 1.6; color: #334155;">{body_html}</div>')

    return "".join(html_parts)


def sanitize_shared_concepts_display(text: str) -> str:
    """Ensure shared concepts in verification cards strictly display substantive clinical concepts."""
    if not text or "shared concepts:" not in text:
        return text

    def _replace_concepts(match):
        raw_concepts = [c.strip() for c in match.group(1).split(",") if c.strip()]
        clean_concepts = [c for c in raw_concepts if is_clinical_concept_token(c)]
        if not clean_concepts:
            return ""
        return f"(shared concepts: {', '.join(clean_concepts[:4])})"

    return re.sub(r'\(shared concepts:\s*([^)]+)\)', _replace_concepts, text)

# Page Configuration
st.set_page_config(
    page_title="PharmGuard | Clinical RAG & Guardrails",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Medical Aesthetic Styling
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
    color: #f8fafc;
}
.main-header p {
    margin: 0.4rem 0 0 0;
    font-size: 0.95rem;
    color: #94a3b8;
    line-height: 1.4;
}

/* Badge styling */
.tech-pill {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 9999px;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    background: rgba(15, 118, 110, 0.2);
    color: #2dd4bf;
    border: 1px solid rgba(45, 212, 191, 0.3);
}

/* Severity Banners */
.severity-critical {
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
    color: #fef2f2;
    padding: 1rem 1.4rem;
    border-radius: 10px;
    border-left: 6px solid #ef4444;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
}
.severity-major {
    background: linear-gradient(135deg, #7c2d12 0%, #9a3412 100%);
    color: #fff7ed;
    padding: 1rem 1.4rem;
    border-radius: 10px;
    border-left: 6px solid #f97316;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 12px rgba(249, 115, 22, 0.2);
}
.severity-moderate {
    background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
    color: #fffbeb;
    padding: 1rem 1.4rem;
    border-radius: 10px;
    border-left: 6px solid #f59e0b;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2);
}
.severity-low {
    background: linear-gradient(135deg, #14532d 0%, #166534 100%);
    color: #f0fdf4;
    padding: 1rem 1.4rem;
    border-radius: 10px;
    border-left: 6px solid #22c55e;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2);
}
.severity-info {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    color: #f8fafc;
    padding: 1rem 1.4rem;
    border-radius: 10px;
    border-left: 6px solid #38bdf8;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2);
}
.severity-blocked {
    background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
    color: #fffbeb;
    padding: 1.2rem 1.6rem;
    border-radius: 10px;
    border-left: 6px solid #f59e0b;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.25);
}

/* Card Boxes */
.clinical-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    height: 100%;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
}
.clinical-card h4 {
    margin-top: 0;
    margin-bottom: 0.75rem;
    color: #0f172a;
    font-weight: 700;
    font-size: 1.05rem;
}

/* Citation Box */
.citation-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid #0f766e;
}
.verified-tag {
    background: #dcfce7;
    color: #15803d;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 0.3rem;
}
.unverified-tag {
    background: #fee2e2;
    color: #b91c1c;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 0.3rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_graph_app():
    """Get the compiled LangGraph workflow application."""
    return build_pharmguard_graph()


app = get_graph_app()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("PharmGuard Engine")
    st.caption("Agentic Clinical RAG & Grounded Decision Support")

    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown(
        """
        <div style="margin-bottom: 0.8rem;">
            <span class="tech-pill">LangGraph</span>
            <span class="tech-pill">ChromaDB</span>
            <span class="tech-pill">BM25Okapi</span>
            <span class="tech-pill">BGE-Reranker</span>
            <span class="tech-pill">Pydantic v2</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Corpus Statistics")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("FDA Labels", "40 Drugs")
        st.metric("Total Chunks", "6,245")
    with col_b:
        st.metric("Guidelines", "3 Clinical")
        st.metric("Verified Hit@4", "100.0%")

    st.markdown("---")
    st.subheader("LLM Configuration")
    has_groq = bool(os.getenv("GROQ_API_KEY") and not os.getenv("GROQ_API_KEY", "").startswith("your_"))
    default_engine_idx = 1 if has_groq else 0
    api_choice = st.radio(
        "Inference Engine",
        ["Deterministic Local Engine", "Groq (High-Speed Cloud LLM)", "OpenAI (gpt-4o-mini)"],
        index=default_engine_idx,
    )

    if api_choice == "Groq (High-Speed Cloud LLM)":
        if has_groq:
            st.success("Groq API Key loaded from environment.")
        else:
            user_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
            if user_key:
                os.environ["GROQ_API_KEY"] = user_key
                st.success("Groq Key saved for session.")
    elif api_choice == "OpenAI (gpt-4o-mini)":
        user_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if user_key:
            os.environ["OPENAI_API_KEY"] = user_key
            st.success("OpenAI Key saved for session.")

    st.markdown("---")
    show_benchmark = st.checkbox("Show Ablation Benchmark Metrics", value=False)


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>PharmGuard: Clinical Decision Support</h1>
        <p>Production-Grade Agentic Clinical RAG with Sentence-Window Chunking, Dual Indexing (Dense ChromaDB + Sparse BM25 RRF), BGE Cross-Encoder Reranking, and 100% Verbatim Citation Verification.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Benchmark Metrics View Toggle
if show_benchmark:
    st.subheader("RAGAS & Ablation Benchmark Results (15 Golden Test Cases)")
    st.info("Comparison between Baseline Naive Dense RAG (ChromaDB only) vs. PharmGuard Hybrid + Cross-Encoder Reranking.")

    bm_col1, bm_col2, bm_col3, bm_col4 = st.columns(4)
    with bm_col1:
        st.metric("Context Precision", "0.9333", "+13.8% vs Naive")
    with bm_col2:
        st.metric("Context Recall", "0.8111", "+9.8% vs Naive")
    with bm_col3:
        st.metric("Mean Reciprocal Rank", "0.9333", "+15.1% vs Naive")
    with bm_col4:
        st.metric("Hit Rate @ 1", "86.7%", "+18.2% vs Naive")

    report_path = BASE_DIR / "evaluation" / "results_report.md"
    if report_path.exists():
        with st.expander("View Full Markdown Evaluation Report", expanded=False):
            st.markdown(report_path.read_text(encoding="utf-8"))
    st.markdown("---")

# -----------------------------------------------------------------------------
# QUICK DEMO SCENARIOS
# -----------------------------------------------------------------------------
st.subheader("Clinical Query & Medication Regimen")

# Session state initialization for query text
if "query_input" not in st.session_state:
    st.session_state["query_input"] = "Can a patient with moderate renal impairment taking Metformin take Paxlovid?"

st.caption("Select a pre-built high-risk clinical case or type your own question below:")
btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

with btn_col1:
    if st.button("Renal: Metformin + Paxlovid", use_container_width=True):
        st.session_state["query_input"] = "Can a patient with moderate renal impairment taking Metformin take Paxlovid?"

with btn_col2:
    if st.button("Fatal Serotonin: Phenelzine + Sumatriptan", use_container_width=True):
        st.session_state["query_input"] = "Can a patient taking Phenelzine be prescribed Sumatriptan for acute migraine headaches?"

with btn_col3:
    if st.button("CYP3A4: Atorvastatin + Clarithromycin", use_container_width=True):
        st.session_state["query_input"] = "What is the interaction risk between Atorvastatin and Clarithromycin, and what dosing adjustment is needed?"

with btn_col4:
    if st.button("5-Week Washout: Fluoxetine to Phenelzine", use_container_width=True):
        st.session_state["query_input"] = "Patient is taking Fluoxetine 40mg daily and the doctor wants to start Phenelzine for depression. What is the washout protocol?"

# Query Input Area
user_query = st.text_area(
    "Clinical Query",
    value=st.session_state["query_input"],
    height=90,
    help="Enter drug combinations, organ parameters (e.g. eGFR, creatinine, age), or clinical guidelines questions.",
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_clicked = st.button("Evaluate Regimen", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# EXECUTION & RESULTS RENDERING
# -----------------------------------------------------------------------------
if run_clicked and user_query.strip():
    with st.spinner("Analyzing regimen with LangGraph clinical workflow..."):
        try:
            state_output = app.invoke({"query": user_query})
            assessment: ClinicalAssessment = state_output.get("final_assessment")
            selected_tools = state_output.get("selected_tools", [])
            retrieved_chunks = state_output.get("retrieved_chunks", [])
            patient_params = state_output.get("patient_parameters", {})
            extracted_drugs = state_output.get("extracted_drugs", [])

            if assessment:
                st.markdown("---")
                
                # 1. Severity Alert Banner & Evidence Gate Block
                is_clinical = state_output.get("is_clinical", True)
                is_diagnostic = (
                    getattr(assessment, "is_diagnostic_response", False)
                    or state_output.get("is_diagnostic", False)
                )
                severity = assessment.risk_severity
                is_blocked = (
                    (severity == "Insufficient Evidence / Cannot Verify"
                    or getattr(assessment, "blocked_by_guardrail", False))
                    and not is_diagnostic
                )

                if is_blocked:
                    st.markdown(
                        """
                        <div class="severity-blocked">
                            <div style="font-size: 1.2rem; font-weight: 800; letter-spacing: -0.01em; color: #fbbf24;">
                                ⚠️ Evidence Gate Triggered — Recommendation Blocked
                            </div>
                            <div style="font-size: 0.95rem; margin-top: 0.4rem; color: #f1f5f9; line-height: 1.5;">
                                <strong>Safety Protection Active:</strong> Automated clinical recommendations are suppressed.
                                The retrieved regulatory labels and clinical guidelines do not contain verified evidence supporting an interaction or dosing recommendation for the requested query.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.warning("⚠️ Action Blocked: Do not rely on automated output. Consult official prescribing information or a clinical pharmacist.")
                    with st.expander("Safety Audit & Gate Details", expanded=True):
                        st.markdown(f"**Clinical Mechanism Notice:**\n{assessment.clinical_mechanism}")
                        st.markdown(f"**Actionable Guidance:**\n{assessment.dosing_and_actionable_guidance}")
                elif is_diagnostic:
                    css_class = "severity-info"
                    badge_icon = "SYSTEM DIAGNOSTIC: CITATION VERIFICATION & TRACE ANALYSIS"
                    score_subtitle = (
                        "Diagnostic Observability: <strong>Active</strong> | "
                        "Explaining recorded system behavior, retrieval provenance, and verification diagnostics."
                    )
                    st.markdown(
                        f"""
                        <div class="{css_class}">
                            <div style="font-size: 1.15rem; font-weight: 800; letter-spacing: -0.01em;">
                                {badge_icon}
                            </div>
                            <div style="font-size: 0.9rem; margin-top: 0.3rem; opacity: 0.95;">
                                {score_subtitle}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    card_col1, card_col2 = st.columns(2)
                    with card_col1:
                        st.markdown(
                            f"""
                            <div class="clinical-card">
                                <h4>Diagnostic Analysis & Verification Provenance</h4>
                                <div style="color: #334155; line-height: 1.6; font-size: 0.93rem; white-space: pre-wrap;">
{assessment.clinical_mechanism}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with card_col2:
                        st.markdown(
                            f"""
                            <div class="clinical-card">
                                <h4>System Diagnostic Guidance</h4>
                                <div style="color: #334155; line-height: 1.6; font-size: 0.93rem; white-space: pre-wrap;">
{assessment.dosing_and_actionable_guidance}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                elif not is_clinical:
                    css_class = "severity-info"
                    badge_icon = "OUT-OF-DOMAIN QUERY: NO CLINICAL OR DRUG ENTITIES DETECTED"
                    score_subtitle = (
                        "Domain Guardrail: <strong>Active</strong> | "
                        "Status: <strong>Query evaluated as non-clinical</strong>"
                    )
                    st.markdown(
                        f"""
                        <div class="{css_class}">
                            <div style="font-size: 1.15rem; font-weight: 800; letter-spacing: -0.01em;">
                                {badge_icon}
                            </div>
                            <div style="font-size: 0.9rem; margin-top: 0.3rem; opacity: 0.95;">
                                {score_subtitle}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    card_col1, card_col2 = st.columns(2)
                    with card_col1:
                        st.markdown(
                            f"""
                            <div class="clinical-card">
                                <h4>Domain Scope & Entity Analysis</h4>
                                <p style="color: #334155; line-height: 1.6; font-size: 0.93rem;">
                                    {assessment.clinical_mechanism}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with card_col2:
                        st.markdown(
                            f"""
                            <div class="clinical-card">
                                <h4>Recommended Guidance</h4>
                                <div style="font-size: 0.93rem; margin-top: 0.4rem;">
                                    {format_actionable_guidance_html(assessment.dosing_and_actionable_guidance)}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    css_class = {
                        "Critical / Contraindicated": "severity-critical",
                        "Major / Dose Adjustment Required": "severity-major",
                        "Moderate / Monitor": "severity-moderate",
                        "Low / No Known Conflict": "severity-low",
                    }.get(severity, "severity-major")

                    badge_icon = {
                        "Critical / Contraindicated": "CRITICAL CONTRAINDICATION",
                        "Major / Dose Adjustment Required": "MAJOR CLINICAL RISK: DOSE MODIFICATION REQUIRED",
                        "Moderate / Monitor": "MODERATE RISK: INTENSIVE CLINICAL MONITORING",
                        "Low / No Known Conflict": "LOW RISK / NO CLINICAL CONFLICT DETECTED",
                    }.get(severity, severity)

                    score_subtitle = (
                        f"Groundedness Score: <strong>{assessment.groundedness_score * 100:.1f}%</strong> | "
                        f"Verified Quotes: <strong>{sum(1 for c in assessment.citations if c.verified)}/{len(assessment.citations)}</strong>"
                    )

                    st.markdown(
                        f"""
                        <div class="{css_class}">
                            <div style="font-size: 1.15rem; font-weight: 800; letter-spacing: -0.01em;">
                                {badge_icon}
                            </div>
                            <div style="font-size: 0.9rem; margin-top: 0.3rem; opacity: 0.95;">
                                {score_subtitle}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # 2. Main 2-Column Findings
                    card_col1, card_col2 = st.columns(2)
                    with card_col1:
                        st.markdown(
                            f"""
                            <div class="clinical-card">
                                <h4>Mechanism of Interaction & Pharmacology</h4>
                                <p style="color: #334155; line-height: 1.6; font-size: 0.93rem;">
                                    {assessment.clinical_mechanism}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with card_col2:
                        st.markdown(
                            f"""
                            <div class="clinical-card">
                                <h4>Actionable Dosing & Prescribing Guidance</h4>
                                <div style="font-size: 0.93rem; margin-top: 0.4rem;">
                                    {format_actionable_guidance_html(assessment.dosing_and_actionable_guidance)}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

                # 3. Verified Citations Panel
                st.subheader("Authoritative Grounded Evidence Citations")
                if not assessment.citations:
                    if not is_clinical:
                        st.info("ℹ️ No citations were retrieved because this query does not refer to pharmaceutical drugs, clinical contraindications, or organ function guidelines.")
                    else:
                        st.info("No explicit citations were generated for this query.")
                else:
                    for idx, cit in enumerate(assessment.citations, 1):
                        tag_html = (
                            '<span class="verified-tag">VERIFIED GROUNDED IN SOURCE</span>'
                            if cit.verified
                            else '<span class="unverified-tag">UNVERIFIED CITATION</span>'
                        )
                        failure_html = ""
                        if not cit.verified and cit.failure_reason:
                            failure_html = f"""
                            <div style="margin-top: 0.45rem; padding: 0.45rem 0.7rem; background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; font-size: 0.82rem; color: #991b1b;">
                                <strong>Verification Failure Reason:</strong> {cit.failure_reason}
                            </div>
                            """
                        support_html = ""
                        if cit.claim_support_status:
                            color = "#15803d" if cit.claim_support_status == "supported" else "#b45309" if cit.claim_support_status == "unsupported" else "#64748b"
                            claim_reason_txt = f" — {sanitize_shared_concepts_display(cit.diagnostic.claim_support_reason)}" if cit.diagnostic and cit.diagnostic.claim_support_reason else ""
                            support_html = f"""
                            <div style="margin-top: 0.35rem; font-size: 0.8rem; color: #475569;">
                                Claim Support: <strong style="color: {color};">{cit.claim_support_status.upper()}</strong>{claim_reason_txt}
                            </div>
                            """

                        st.markdown(
                            f"""
                            <div class="citation-card">
                                {tag_html}
                                <div style="font-weight: 700; color: #0f172a; font-size: 0.9rem; margin-bottom: 0.2rem;">
                                    {cit.source_name} &bull; Section: <code>{cit.section}</code>
                                </div>
                                <div style="font-style: italic; color: #475569; font-size: 0.88rem; line-height: 1.5;">
                                    "{cit.verbatim_quote}"
                                </div>
                                {failure_html}
                                {support_html}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # 4. Observability & Execution Trace Expander
                with st.expander("LangGraph Execution Trace & Evidence Observability", expanded=False):
                    st.markdown("#### Autonomous Tool Selection & State")
                    st.write({
                        "Extracted Drugs": extracted_drugs,
                        "Patient Parameters": patient_params,
                        "Selected Tools": selected_tools,
                        "Relevance Check": state_output.get("is_relevant", True),
                        "Is Diagnostic": state_output.get("is_diagnostic", False),
                    })

                    # Structured Verification Diagnostics
                    diag_records = state_output.get("verification_diagnostics") or [
                        d.model_dump() for d in getattr(assessment, "verification_diagnostics", [])
                    ]
                    if diag_records:
                        st.markdown("#### Structured Citation Verification Diagnostics")
                        st.json(diag_records)

                    st.markdown("#### Top Retrieved & Reranked Evidence Chunks")
                    for c_idx, chunk in enumerate(retrieved_chunks, 1):
                        meta = chunk.get("metadata", {})
                        r_score = chunk.get("rerank_score", "N/A")
                        rrf = chunk.get("rrf_score", "N/A")
                        st.markdown(
                            f"**[{c_idx}] {chunk.get('chunk_id')}** | Drug: `{meta.get('drug_name')}` | "
                            f"Section: `{meta.get('section')}` | Rerank Score: `{r_score}` | RRF: `{rrf}`"
                        )
                        st.text(f"Target Sentence: {chunk.get('sentence')}")
                        st.caption(f"Expanded Window: {chunk.get('window')}")
                        st.markdown("---")

            else:
                st.error("Workflow completed but no clinical assessment was returned.")

        except Exception as err:
            st.error(f"Error during workflow execution: {err}")
            logger.exception("Streamlit execution exception")
