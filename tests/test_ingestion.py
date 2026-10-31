"""Unit tests for Phase 1 data ingestion modules."""

import pytest
from src.ingestion.fda_downloader import (
    DRUGS_OF_INTEREST,
    clean_section,
    extract_clinical_sections,
)
from src.ingestion.pdf_parser import clean_text, table_to_markdown, clean_extracted_table


def test_drugs_of_interest_count():
    """Verify that exactly 40 high-risk drugs are configured with no duplicates."""
    assert len(DRUGS_OF_INTEREST) == 40
    assert len(set(DRUGS_OF_INTEREST)) == 40
    assert "Warfarin" in DRUGS_OF_INTEREST
    assert "Paxlovid" in DRUGS_OF_INTEREST
    assert "Apixaban" in DRUGS_OF_INTEREST
    assert "Aspirin" in DRUGS_OF_INTEREST


def test_clean_section():
    """Test text normalization for openFDA section arrays."""
    assert clean_section(None) == ""
    assert clean_section(["Line 1", "Line 2"]) == "Line 1\n\nLine 2"
    assert clean_section("Single block of text") == "Single block of text"


def test_extract_clinical_sections():
    """Test extraction of clinical sections from raw mock FDA label."""
    mock_raw = {
        "id": "mock-123",
        "boxed_warning": ["WARNING: Risk of severe bleeding."],
        "contraindications": ["Hypersensitivity to warfarin."],
        "drug_interactions": ["Interacts with NSAIDs and CYP2C9 inhibitors."],
        "warnings_and_cautions": ["Monitor INR regularly."],
        "dosage_and_administration": ["Initial dose 2 to 5 mg daily."],
        "openfda": {
            "generic_name": ["WARFARIN SODIUM"],
            "brand_name": ["COUMADIN"],
            "manufacturer_name": ["Bristol-Myers Squibb"],
        },
    }
    extracted = extract_clinical_sections("Warfarin", mock_raw)
    assert extracted["drug_name"] == "Warfarin"
    assert extracted["has_boxed_warning"] is True
    assert "bleeding" in extracted["sections"]["boxed_warning"].lower()
    assert "nsaids" in extracted["sections"]["drug_interactions"].lower()
    assert extracted["generic_name"] == "WARFARIN SODIUM"


def test_clean_text():
    """Test stripping running headers and footers."""
    raw = (
        "Page 1 of 3\n"
        "Reference Code: CPG-RENAL-2026.01\n"
        "KDIGO Guideline for Renal Impairment\n"
        "\n\n\n"
        "Metformin is contraindicated if eGFR < 30 mL/min.\n"
    )
    cleaned = clean_text(raw)
    assert "Page 1 of 3" not in cleaned
    assert "Reference Code:" not in cleaned
    assert "Metformin is contraindicated" in cleaned


def test_table_to_markdown():
    """Test formatting raw table cells into GitHub-flavored Markdown."""
    headers = ["Drug", "Threshold", "Action"]
    rows = [
        ["Metformin", "eGFR < 30", "Contraindicated"],
        ["Apixaban", "Cr >= 1.5", "Reduce dose to 2.5mg BID"],
    ]
    md = table_to_markdown(headers, rows)
    assert "| Drug | Threshold | Action |" in md
    assert "| Metformin | eGFR < 30 | Contraindicated |" in md
