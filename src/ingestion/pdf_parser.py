"""Clinical Guideline PDF Parser for PharmGuard.

Extracts text, structured tables, and clinical metadata from multi-page guideline
PDFs using pdfplumber. Cleans repetitive headers/footers and normalizes tables
into structured JSON and Markdown formats for hybrid retrieval.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pdfplumber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pdf_parser")

GUIDELINES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "guidelines"
OUTPUT_FILE = GUIDELINES_DIR / "parsed_guidelines.json"

# Header/Footer patterns to strip during cleaning
HEADER_FOOTER_PATTERNS = [
    re.compile(r"^Page \d+( of \d+)?$", re.IGNORECASE),
    re.compile(r"^Reference Code: CPG-.*$", re.IGNORECASE),
    re.compile(r"^CONFIDENTIAL & CLINICAL USE ONLY.*$", re.IGNORECASE),
]


def clean_text(raw_text: Optional[str]) -> str:
    """Clean page text by stripping running headers, footers, and redundant whitespace."""
    if not raw_text:
        return ""

    cleaned_lines = []
    lines = raw_text.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check against header/footer regexes
        if any(pat.match(stripped) for pat in HEADER_FOOTER_PATTERNS):
            continue

        cleaned_lines.append(stripped)

    # Join lines and collapse multiple consecutive newlines
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def table_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """Format structured table rows into standard Markdown table."""
    if not headers and not rows:
        return ""

    if not headers and rows:
        headers = [f"Col {i+1}" for i in range(len(rows[0]))]

    # Clean cells of newlines for markdown table row integrity
    safe_headers = [str(h).replace("\n", " ").strip() for h in headers]
    md_lines = [
        "| " + " | ".join(safe_headers) + " |",
        "| " + " | ".join(["---"] * len(safe_headers)) + " |",
    ]

    for row in rows:
        # Match column length
        safe_cells = []
        for i in range(len(safe_headers)):
            cell_val = row[i] if i < len(row) else ""
            clean_cell = str(cell_val or "").replace("\n", " ").replace("|", "\\|").strip()
            safe_cells.append(clean_cell)
        md_lines.append("| " + " | ".join(safe_cells) + " |")

    return "\n".join(md_lines)


def clean_extracted_table(raw_table: List[List[Optional[str]]]) -> Dict[str, Any]:
    """Clean and normalize a raw table extracted by pdfplumber."""
    if not raw_table or len(raw_table) == 0:
        return {"headers": [], "rows": [], "markdown": ""}

    cleaned_table = []
    for row in raw_table:
        if not row or not any(row):
            continue
        cleaned_row = [(cell or "").strip() for cell in row]
        if any(cleaned_row):
            cleaned_table.append(cleaned_row)

    if not cleaned_table:
        return {"headers": [], "rows": [], "markdown": ""}

    # First row is typically header
    headers = cleaned_table[0]
    data_rows = cleaned_table[1:] if len(cleaned_table) > 1 else []

    md = table_to_markdown(headers, data_rows)

    return {
        "headers": headers,
        "rows": data_rows,
        "row_count": len(data_rows),
        "col_count": len(headers),
        "markdown": md,
    }


def parse_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Extract full text, metadata, and structured tables from a single clinical guideline PDF."""
    logger.info("Parsing PDF: %s", pdf_path.name)
    pages_data = []
    all_tables = []
    full_text_parts = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        
        for idx, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text(layout=True) or ""
            cleaned_ptext = clean_text(raw_text)
            
            raw_tables = page.extract_tables() or []
            page_tables = []
            
            for t_idx, raw_table in enumerate(raw_tables):
                cleaned_tbl = clean_extracted_table(raw_table)
                if cleaned_tbl["headers"]:
                    cleaned_tbl["table_id"] = f"{pdf_path.stem}_p{idx}_t{t_idx+1}"
                    page_tables.append(cleaned_tbl)
                    all_tables.append(cleaned_tbl)
            
            pages_data.append({
                "page_number": idx,
                "text": cleaned_ptext,
                "tables": page_tables,
                "table_count": len(page_tables),
            })
            
            if cleaned_ptext:
                full_text_parts.append(cleaned_ptext)
                
            # If tables exist, also append their markdown representations to text for indexers
            for tbl in page_tables:
                if tbl.get("markdown"):
                    full_text_parts.append(f"\n[TABLE: {tbl['table_id']}]\n{tbl['markdown']}\n")

    # Inferred title from filename
    title = pdf_path.stem.replace("_", " ").title()

    parsed_guideline = {
        "guideline_id": pdf_path.stem,
        "filename": pdf_path.name,
        "title": title,
        "total_pages": total_pages,
        "total_tables": len(all_tables),
        "pages": pages_data,
        "all_tables": all_tables,
        "full_text": "\n\n".join(full_text_parts),
    }

    logger.info(
        "Finished '%s': %d pages, %d tables, %d total chars",
        pdf_path.name,
        total_pages,
        len(all_tables),
        len(parsed_guideline["full_text"]),
    )
    return parsed_guideline


def parse_all_guidelines(guidelines_dir: Path = GUIDELINES_DIR, output_file: Path = OUTPUT_FILE) -> Dict[str, Any]:
    """Parse all clinical guideline PDFs in the directory and save structured output."""
    pdf_files = sorted(list(guidelines_dir.glob("*.pdf")))
    if not pdf_files:
        logger.warning("No PDF files found in %s", guidelines_dir)
        return {"total_guidelines": 0, "guidelines": []}

    logger.info("Found %d PDF guidelines in %s", len(pdf_files), guidelines_dir)
    parsed_docs = []

    for pdf_file in pdf_files:
        parsed_doc = parse_pdf(pdf_file)
        parsed_docs.append(parsed_doc)

    output_payload = {
        "total_guidelines": len(parsed_docs),
        "guidelines": parsed_docs,
        "guideline_ids": [doc["guideline_id"] for doc in parsed_docs],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    logger.info("Successfully wrote parsed clinical guidelines to %s", output_file)
    return output_payload


if __name__ == "__main__":
    parse_all_guidelines()
