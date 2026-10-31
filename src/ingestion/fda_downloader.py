"""FDA Drug Label Downloader for PharmGuard.

Queries the public openFDA API for 40 high-risk interaction drugs and extracts
critical clinical safety sections:
  - Boxed Warning
  - Contraindications
  - Drug Interactions
  - Warnings and Precautions
  - Dosage and Administration
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fda_downloader")

# 40 High-Risk Interaction Drugs Specified for PharmGuard
DRUGS_OF_INTEREST: List[str] = [
    "Warfarin",
    "Apixaban",
    "Atorvastatin",
    "Lisinopril",
    "Amiodarone",
    "Digoxin",
    "Clopidogrel",
    "Spironolactone",
    "Metformin",
    "Empagliflozin",
    "Semaglutide",
    "Glimepiride",
    "Levothyroxine",
    "Paxlovid",
    "Ciprofloxacin",
    "Azithromycin",
    "Fluconazole",
    "Clarithromycin",
    "Rifampin",
    "Metronidazole",
    "Sertraline",
    "Fluoxetine",
    "Phenelzine",
    "Lithium",
    "Sumatriptan",
    "Alprazolam",
    "Bupropion",
    "Venlafaxine",
    "Ibuprofen",
    "Methotrexate",
    "Cyclosporine",
    "Tacrolimus",
    "Prednisone",
    "Tramadol",
    "Oxycodone",
    "Colchicine",
    "Allopurinol",
    "Azathioprine",
    "Celecoxib",
    "Aspirin",
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "fda_labels"

# Special search aliases for combination drugs or naming variants in openFDA
DRUG_QUERY_ALIASES = {
    "Paxlovid": [
        'openfda.brand_name:"PAXLOVID"',
        'openfda.substance_name:"NIRMATRELVIR"',
        'PAXLOVID',
    ],
    "Levothyroxine": [
        'openfda.generic_name:"LEVOTHYROXINE SODIUM"',
        'openfda.generic_name:"LEVOTHYROXINE"',
        'openfda.brand_name:"SYNTHROID"',
    ],
    "Aspirin": [
        'openfda.generic_name:"ASPIRIN"',
        'openfda.substance_name:"ASPIRIN"',
        'openfda.brand_name:"ASPIRIN"',
    ],
}


def clean_section(content: Any) -> str:
    """Normalize label sections which can be lists of strings or raw strings."""
    if not content:
        return ""
    if isinstance(content, list):
        return "\n\n".join(str(item).strip() for item in content if item)
    return str(content).strip()


def query_openfda(query_str: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """Query openFDA drug label API endpoint."""
    base_url = "https://api.fda.gov/drug/label.json"
    params = {"search": query_str, "limit": 1}
    try:
        response = requests.get(base_url, params=params, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                return results[0]
        elif response.status_code == 404:
            # Query matched 0 results
            return None
        elif response.status_code == 429:
            logger.warning("openFDA rate limit encountered. Waiting 5 seconds...")
            time.sleep(5)
        else:
            logger.debug("openFDA returned status %d for query %s", response.status_code, query_str)
    except Exception as exc:
        logger.warning("Request error for query '%s': %s", query_str, exc)
    return None


def fetch_drug_label(drug: str) -> Optional[Dict[str, Any]]:
    """Fetch FDA label for a given drug using progressive query strategies."""
    # Build candidate queries
    queries = []
    
    # Check custom aliases first
    if drug in DRUG_QUERY_ALIASES:
        queries.extend(DRUG_QUERY_ALIASES[drug])
        
    # Standard query strategies
    drug_lower = drug.lower()
    queries.extend([
        f'openfda.generic_name:"{drug_lower}"+openfda.brand_name:"{drug_lower}"',
        f'openfda.generic_name:"{drug_lower}"',
        f'openfda.brand_name:"{drug_lower}"',
        f'openfda.substance_name:"{drug_lower}"',
        f'"{drug_lower}"',
    ])
    
    for q in queries:
        result = query_openfda(q)
        if result:
            logger.info("Found FDA label for '%s' using query: %s", drug, q)
            return result
        time.sleep(0.3)
        
    return None


def extract_clinical_sections(drug: str, raw_label: Dict[str, Any]) -> Dict[str, Any]:
    """Extract targeted clinical sections and openFDA metadata."""
    openfda_meta = raw_label.get("openfda", {})
    
    # Section extractors with fallbacks
    boxed_warning = clean_section(
        raw_label.get("boxed_warning")
        or raw_label.get("boxed_warnings")
        or raw_label.get("black_box_warning")
    )
    contraindications = clean_section(
        raw_label.get("contraindications")
        or raw_label.get("contraindications_table")
    )
    drug_interactions = clean_section(
        raw_label.get("drug_interactions")
        or raw_label.get("drug_interactions_table")
    )
    warnings_and_cautions = clean_section(
        raw_label.get("warnings_and_cautions")
        or raw_label.get("warnings_and_cautions_table")
        or raw_label.get("warnings")
        or raw_label.get("precautions")
    )
    dosage_and_administration = clean_section(
        raw_label.get("dosage_and_administration")
        or raw_label.get("dosage_and_administration_table")
    )
    
    # Collect brand names & generic names
    brand_names = openfda_meta.get("brand_name", [])
    generic_names = openfda_meta.get("generic_name", [])
    substance_names = openfda_meta.get("substance_name", [])
    pharm_classes = openfda_meta.get("pharm_class_cs", []) or openfda_meta.get("pharm_class_epc", [])

    return {
        "drug_name": drug,
        "generic_name": generic_names[0] if generic_names else drug,
        "brand_names": brand_names,
        "all_generic_names": generic_names,
        "substance_names": substance_names,
        "pharmaceutical_classes": pharm_classes,
        "manufacturer": (openfda_meta.get("manufacturer_name", ["Unknown"]) or ["Unknown"])[0],
        "spl_id": (openfda_meta.get("spl_id", [""]) or [""])[0],
        "set_id": raw_label.get("set_id", ""),
        "id": raw_label.get("id", ""),
        "effective_time": raw_label.get("effective_time", ""),
        "sections": {
            "boxed_warning": boxed_warning,
            "contraindications": contraindications,
            "drug_interactions": drug_interactions,
            "warnings_and_cautions": warnings_and_cautions,
            "dosage_and_administration": dosage_and_administration,
        },
        "has_boxed_warning": bool(boxed_warning),
        "source": "openFDA Drug Label API",
    }


def download_all_drug_labels(output_dir: Path = OUTPUT_DIR) -> Dict[str, Any]:
    """Download labels for all 40 drugs and save JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total": len(DRUGS_OF_INTEREST),
        "success": 0,
        "failed": 0,
        "with_boxed_warning": 0,
        "failed_drugs": [],
    }
    
    logger.info("Starting ingestion of %d high-risk drugs to %s", len(DRUGS_OF_INTEREST), output_dir)
    
    for idx, drug in enumerate(DRUGS_OF_INTEREST, 1):
        target_file = output_dir / f"{drug}.json"
        logger.info("[%d/%d] Fetching FDA label for '%s'...", idx, len(DRUGS_OF_INTEREST), drug)
        
        raw_label = fetch_drug_label(drug)
        if not raw_label:
            logger.error("Failed to retrieve FDA label for drug: '%s'", drug)
            stats["failed"] += 1
            stats["failed_drugs"].append(drug)
            continue
            
        parsed_data = extract_clinical_sections(drug, raw_label)
        
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            
        stats["success"] += 1
        if parsed_data["has_boxed_warning"]:
            stats["with_boxed_warning"] += 1
            
        logger.info(
            "Saved %s (Boxed Warning: %s, Interactions length: %d chars)",
            target_file.name,
            "YES" if parsed_data["has_boxed_warning"] else "No",
            len(parsed_data["sections"]["drug_interactions"]),
        )
        
        # Friendly rate pacing
        time.sleep(0.4)
        
    logger.info("Ingestion completed! Summary: %s", stats)
    return stats


if __name__ == "__main__":
    download_all_drug_labels()
