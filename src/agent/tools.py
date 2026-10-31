"""Autonomous tools for the PharmGuard clinical reasoning agent.

Includes:
1. search_fda_and_guidelines: HybridRetriever + CrossEncoder reranking
2. renal_egfr_calculator_tool: Cockcroft-Gault CrCl calculation and KDIGO CKD staging
3. openfda_live_fallback_tool: Dynamic live openFDA lookup for unindexed drugs
"""

import re
import logging
from typing import List, Dict, Any, Optional

import requests
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import ClinicalReranker
from src.ingestion.fda_downloader import (
    fetch_drug_label,
    extract_clinical_sections,
    DRUGS_OF_INTEREST,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent_tools")

# Global singletons for retriever and reranker to avoid reloading models on each turn
_RETRIEVER_INSTANCE: Optional[HybridRetriever] = None
_RERANKER_INSTANCE: Optional[ClinicalReranker] = None


def get_retriever() -> HybridRetriever:
    """Get or create singleton HybridRetriever."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        logger.info("Initializing singleton HybridRetriever for agent tools...")
        _RETRIEVER_INSTANCE = HybridRetriever()
    return _RETRIEVER_INSTANCE


def get_reranker() -> ClinicalReranker:
    """Get or create singleton ClinicalReranker."""
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        logger.info("Initializing singleton ClinicalReranker for agent tools...")
        _RERANKER_INSTANCE = ClinicalReranker()
    return _RERANKER_INSTANCE


def search_fda_and_guidelines(
    query: str,
    top_n: int = 4,
    hybrid_candidates: int = 15,
    extracted_drugs: Optional[List[str]] = None,
    clinical_terms: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Execute hybrid dense + sparse retrieval followed by cross-encoder reranking.

    Args:
        query: Clinical search query.
        top_n: Number of top reranked chunks to return (default 4).
        hybrid_candidates: Number of candidates to gather from RRF (default 15).
        extracted_drugs: Optional list of target canonical drugs.
        clinical_terms: Optional list of queried clinical parameters.

    Returns:
        List of top_n reranked clinical chunks with scores and window context.
    """
    retriever = get_retriever()
    reranker = get_reranker()

    candidates = retriever.retrieve(
        query,
        top_k=hybrid_candidates,
        target_drugs=extracted_drugs,
        clinical_terms=clinical_terms,
    )
    if not candidates:
        logger.warning("No candidates found for query: '%s'", query)
        return []

    top_reranked = reranker.rerank(
        query,
        candidates,
        extracted_drugs=extracted_drugs,
        clinical_terms=clinical_terms,
        top_n=top_n,
    )
    return top_reranked


def renal_egfr_calculator_tool(
    age: Optional[float] = None,
    weight_kg: Optional[float] = None,
    serum_creatinine_mg_dl: Optional[float] = None,
    is_female: bool = False,
    egfr_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate Cockcroft-Gault Creatinine Clearance (CrCl) and KDIGO CKD Stage.

    Args:
        age: Patient age in years.
        weight_kg: Patient weight in kilograms.
        serum_creatinine_mg_dl: Serum creatinine in mg/dL.
        is_female: Whether patient is female (0.85 factor for CrCl).
        egfr_value: Direct eGFR value in mL/min/1.73m² (if CrCl parameters not provided).

    Returns:
        Dictionary containing CrCl, calculated eGFR, KDIGO CKD stage, and clinical warnings.
    """
    crcl: Optional[float] = None
    stage: str = "Unknown"
    category: str = "Unknown"
    clinical_alerts: List[str] = []

    # 1. Calculate Cockcroft-Gault CrCl if parameters available
    if age is not None and weight_kg is not None and serum_creatinine_mg_dl is not None:
        if serum_creatinine_mg_dl > 0 and age > 0 and weight_kg > 0:
            crcl_val = ((140.0 - age) * weight_kg) / (72.0 * serum_creatinine_mg_dl)
            if is_female:
                crcl_val *= 0.85
            crcl = round(crcl_val, 1)

    # Effective filtration metric for staging
    effective_gfr = crcl if crcl is not None else egfr_value

    if effective_gfr is not None:
        if effective_gfr >= 90:
            stage = "Stage G1"
            category = "Normal or high filtration (≥90 mL/min)"
        elif effective_gfr >= 60:
            stage = "Stage G2"
            category = "Mildly decreased renal function (60–89 mL/min)"
        elif effective_gfr >= 45:
            stage = "Stage G3a"
            category = "Mild-to-moderate decrease in renal function (45–59 mL/min)"
        elif effective_gfr >= 30:
            stage = "Stage G3b"
            category = "Moderate-to-severe decrease in renal function (30–44 mL/min)"
        elif effective_gfr >= 15:
            stage = "Stage G4"
            category = "Severely decreased renal function (15–29 mL/min)"
        else:
            stage = "Stage G5"
            category = "Kidney failure / End-Stage Renal Disease (<15 mL/min)"

        # Clinical rule triggers
        if effective_gfr < 30:
            clinical_alerts.append("METFORMIN: CONTRAINDICATED (eGFR <30 mL/min). Risk of fatal lactic acidosis.")
            clinical_alerts.append("PAXLOVID: CONTRAINDICATED or requires severe hemodialysis-specific dosing if eGFR <30.")
            clinical_alerts.append("SPIRONOLACTONE: CONTRAINDICATED (eGFR <30 mL/min). Severe hyperkalemia risk.")
            clinical_alerts.append("METHOTREXATE: CONTRAINDICATED (eGFR <30 mL/min). Severe myelosuppression risk.")
        elif effective_gfr < 60:
            clinical_alerts.append("PAXLOVID: Dose reduction required for moderate impairment (30-59 mL/min): 150mg/100mg BID.")
            clinical_alerts.append("METFORMIN: Max 1000 mg/day if eGFR 30-44 mL/min; do not initiate.")
            clinical_alerts.append("APIXABAN: Review ABC criteria (Serum Cr ≥1.5, Age ≥80, Wt ≤60kg).")

    return {
        "calculated_crcl_ml_min": crcl,
        "provided_egfr": egfr_value,
        "effective_filtration_rate": effective_gfr,
        "kdigo_stage": stage,
        "clinical_category": category,
        "clinical_alerts": clinical_alerts,
        "summary": (
            f"Renal Assessment: {stage} ({category}). "
            f"Filtration rate: {effective_gfr} mL/min. "
            f"Alerts: {len(clinical_alerts)} safety advisories."
        ),
    }


def openfda_live_fallback_tool(drug_name: str) -> Dict[str, Any]:
    """Query live openFDA API for drugs not present in local 40-drug ChromaDB index.

    Args:
        drug_name: Generic or brand name of the drug.

    Returns:
        Structured label dictionary containing critical clinical sections.
    """
    logger.info("Executing live openFDA fallback query for '%s'...", drug_name)
    raw_label = fetch_drug_label(drug_name)

    if not raw_label:
        logger.warning("Live openFDA API found no records for drug '%s'", drug_name)
        return {
            "found": False,
            "drug_name": drug_name,
            "message": f"No openFDA drug label found for '{drug_name}'.",
            "sections": {},
        }

    parsed = extract_clinical_sections(drug_name, raw_label)
    logger.info("Successfully fetched live FDA label for '%s'", drug_name)

    return {
        "found": True,
        "drug_name": drug_name,
        "generic_name": parsed.get("generic_name", drug_name),
        "brand_names": parsed.get("brand_names", []),
        "sections": parsed.get("sections", {}),
        "has_boxed_warning": parsed.get("has_boxed_warning", False),
        "source": "Live openFDA API Fallback",
    }
