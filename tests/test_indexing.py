"""Unit tests for Phase 2 Sentence-Window Chunking and Indexing."""

import json
from pathlib import Path
import pytest

from src.indexing.chunker import create_sentence_windows, clean_sentence
from src.indexing.store import tokenize_bm25


def test_clean_sentence():
    """Test text cleaning for sentence chunking."""
    raw = "  • Warning: Apixaban increases bleeding risk.   "
    cleaned = clean_sentence(raw)
    assert cleaned == "Warning: Apixaban increases bleeding risk."


def test_sentence_window_construction():
    """Test 3-before + target + 3-after windowing boundaries."""
    sentences = [f"Sentence {i}" for i in range(10)]
    
    chunks = create_sentence_windows(
        sentences=sentences,
        doc_id="test_drug",
        drug_name="TestDrug",
        source_type="fda_label",
        section="boxed_warning",
        window_size=3,
    )
    
    assert len(chunks) == 10
    
    # First chunk (index 0): window should have 4 sentences (0, 1, 2, 3)
    c0 = chunks[0]
    assert c0["sentence"] == "Sentence 0"
    assert c0["window"] == "Sentence 0 Sentence 1 Sentence 2 Sentence 3"
    assert c0["metadata"]["drug_name"] == "TestDrug"
    assert c0["metadata"]["source_type"] == "fda_label"
    assert c0["metadata"]["section"] == "boxed_warning"
    
    # Middle chunk (index 5): window should have 7 sentences (2, 3, 4, 5, 6, 7, 8)
    c5 = chunks[5]
    assert c5["sentence"] == "Sentence 5"
    assert c5["window"] == "Sentence 2 Sentence 3 Sentence 4 Sentence 5 Sentence 6 Sentence 7 Sentence 8"
    
    # Last chunk (index 9): window should have 4 sentences (6, 7, 8, 9)
    c9 = chunks[9]
    assert c9["sentence"] == "Sentence 9"
    assert c9["window"] == "Sentence 6 Sentence 7 Sentence 8 Sentence 9"


def test_tokenize_bm25():
    """Test BM25 tokenization for medical terms and symbols."""
    text = "Paxlovid (nirmatrelvir/ritonavir) inhibits CYP3A4-mediated clearance!"
    tokens = tokenize_bm25(text)
    assert "paxlovid" in tokens
    assert "cyp3a4" in tokens or "cyp3a4-mediated" in tokens
    assert "clearance" in tokens


def test_corpus_chunks_file_integrity():
    """Test that generated corpus_chunks.jsonl has valid schema and records."""
    chunks_file = Path(__file__).resolve().parent.parent / "data" / "processed" / "corpus_chunks.jsonl"
    assert chunks_file.exists(), "corpus_chunks.jsonl does not exist"
    
    with open(chunks_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        assert first_line, "corpus_chunks.jsonl is empty"
        data = json.loads(first_line)
        
        assert "chunk_id" in data
        assert "sentence" in data
        assert "window" in data
        assert "metadata" in data
        assert "drug_name" in data["metadata"]
        assert "source_type" in data["metadata"]
        assert "section" in data["metadata"]
        assert "window" in data["metadata"]
