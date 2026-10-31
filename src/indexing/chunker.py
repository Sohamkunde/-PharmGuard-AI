"""Sentence-Window Chunker for PharmGuard.

Processes raw FDA drug labels and parsed clinical guidelines:
1. Splits medical text into individual sentences using SpaCy (en_core_web_sm).
2. Generates Sentence-Window chunks (3 sentences before + target sentence + 3 sentences after).
3. Attaches rich metadata (drug_name, source_type, section, window).
4. Persists the corpus to data/processed/corpus_chunks.jsonl.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator

import spacy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chunker")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FDA_DIR = BASE_DIR / "data" / "raw" / "fda_labels"
GUIDELINES_FILE = BASE_DIR / "data" / "raw" / "guidelines" / "parsed_guidelines.json"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "corpus_chunks.jsonl"

TARGET_FDA_SECTIONS = [
    "boxed_warning",
    "contraindications",
    "drug_interactions",
    "warnings_and_cautions",
    "dosage_and_administration",
]

# Map guideline identifiers to primary drug/topic names for high-precision metadata
GUIDELINE_TOPIC_MAP = {
    "renal_egfr_dosing_guidelines": "Renal Impairment & eGFR Dosing",
    "cyp3a4_drug_interactions_guideline": "CYP3A4 Drug-Drug Interactions",
    "serotonin_syndrome_clinical_guideline": "Serotonin Syndrome Diagnosis & Management",
}


def load_spacy_model():
    """Load SpaCy pipeline configured for high-efficiency sentence segmentation."""
    try:
        logger.info("Attempting to load SpaCy en_core_web_sm pipeline...")
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
        logger.info("Successfully loaded en_core_web_sm.")
    except Exception as exc:
        logger.warning(
            "SpaCy en_core_web_sm binary deserialization encountered an issue (%s). "
            "Falling back to spacy.blank('en') with sentencizer.",
            exc,
        )
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
    nlp.max_length = 2_500_000
    return nlp


def clean_sentence(text: str) -> str:
    """Normalize whitespace and strip leading bullet/number artifacts."""
    text = " ".join(text.split())
    # Strip standalone bullet chars
    text = text.lstrip("•*- \t")
    return text.strip()


def extract_sentences_from_text(nlp, text: str, min_chars: int = 15) -> List[str]:
    """Split text into cleaned, meaningful clinical sentences."""
    if not text or not text.strip():
        return []

    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        cleaned = clean_sentence(sent.text)
        # Filter out trivial fragments (e.g., page numbers, single isolated codes)
        if len(cleaned) >= min_chars and not cleaned.isdigit():
            sentences.append(cleaned)
    return sentences


def create_sentence_windows(
    sentences: List[str],
    doc_id: str,
    drug_name: str,
    source_type: str,
    section: str,
    window_size: int = 3,
) -> List[Dict[str, Any]]:
    """Build sentence-window chunks with 3 sentences before and 3 sentences after."""
    chunks = []
    total = len(sentences)

    for i, target_sentence in enumerate(sentences):
        # 3 sentences before + target + 3 sentences after
        start_idx = max(0, i - window_size)
        end_idx = min(total, i + window_size + 1)
        window = " ".join(sentences[start_idx:end_idx])

        chunk_id = f"{source_type}_{doc_id}_{section}_{i:04d}"
        metadata = {
            "drug_name": drug_name,
            "source_type": source_type,
            "section": section,
            "window": window,
            "sentence_index": i,
            "total_section_sentences": total,
        }

        chunks.append({
            "chunk_id": chunk_id,
            "sentence": target_sentence,
            "window": window,
            "metadata": metadata,
        })

    return chunks


def process_fda_labels(nlp, fda_dir: Path = FDA_DIR) -> List[Dict[str, Any]]:
    """Process all raw FDA label JSON files into sentence-window chunks."""
    all_chunks = []
    label_files = sorted(list(fda_dir.glob("*.json")))
    logger.info("Processing %d FDA drug label files from %s...", len(label_files), fda_dir)

    for label_file in label_files:
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error("Error reading %s: %s", label_file.name, exc)
            continue

        drug_name = data.get("drug_name") or label_file.stem
        sections = data.get("sections", {})

        for sec_name in TARGET_FDA_SECTIONS:
            sec_text = sections.get(sec_name, "")
            if not sec_text or not sec_text.strip():
                continue

            sentences = extract_sentences_from_text(nlp, sec_text)
            if not sentences:
                continue

            chunks = create_sentence_windows(
                sentences=sentences,
                doc_id=drug_name,
                drug_name=drug_name,
                source_type="fda_label",
                section=sec_name,
                window_size=3,
            )
            all_chunks.extend(chunks)

    logger.info("Generated %d sentence-window chunks from FDA labels.", len(all_chunks))
    return all_chunks


def format_table_row_as_sentence(headers: List[str], row: List[str]) -> str:
    """Convert structured clinical table row into a natural language sentence."""
    parts = []
    for h, val in zip(headers, row):
        h_clean = h.replace("\n", " ").strip()
        val_clean = val.replace("\n", " ").strip()
        if h_clean and val_clean:
            parts.append(f"{h_clean}: {val_clean}")
    return ". ".join(parts) + "."


def process_clinical_guidelines(nlp, guidelines_file: Path = GUIDELINES_FILE) -> List[Dict[str, Any]]:
    """Process parsed clinical guidelines into sentence-window chunks."""
    all_chunks = []
    if not guidelines_file.exists():
        logger.warning("Parsed guidelines file not found at %s", guidelines_file)
        return all_chunks

    logger.info("Processing parsed clinical guidelines from %s...", guidelines_file)
    with open(guidelines_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    guidelines = data.get("guidelines", [])

    for g_idx, guideline in enumerate(guidelines, start=1):
        guideline_id = guideline.get("guideline_id", f"guideline_{g_idx}")
        topic_name = GUIDELINE_TOPIC_MAP.get(guideline_id, guideline.get("title", guideline_id))

        pages = guideline.get("pages", [])
        for page in pages:
            page_num = page.get("page_number", 1)
            page_text = page.get("text", "")
            page_sentences = extract_sentences_from_text(nlp, page_text)

            # Also convert structured tables into descriptive clinical sentences
            tables = page.get("tables", [])
            for tbl in tables:
                headers = tbl.get("headers", [])
                rows = tbl.get("rows", [])
                for row in rows:
                    row_sentence = format_table_row_as_sentence(headers, row)
                    if len(row_sentence) >= 20:
                        page_sentences.append(row_sentence)

            if not page_sentences:
                continue

            chunks = create_sentence_windows(
                sentences=page_sentences,
                doc_id=guideline_id,
                drug_name=topic_name,
                source_type="clinical_guideline",
                section=f"page_{page_num}",
                window_size=3,
            )
            all_chunks.extend(chunks)

    logger.info("Generated %d sentence-window chunks from clinical guidelines.", len(all_chunks))
    return all_chunks


def run_chunking(output_file: Path = OUTPUT_FILE) -> int:
    """Run full sentence-window chunking pipeline and write JSONL."""
    nlp = load_spacy_model()

    fda_chunks = process_fda_labels(nlp)
    guideline_chunks = process_clinical_guidelines(nlp)

    total_chunks = fda_chunks + guideline_chunks
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving %d total chunks to %s...", len(total_chunks), output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in total_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(
        "Successfully wrote %d chunks (%d FDA + %d Guidelines) to %s",
        len(total_chunks),
        len(fda_chunks),
        len(guideline_chunks),
        output_file,
    )
    return len(total_chunks)


if __name__ == "__main__":
    run_chunking()
