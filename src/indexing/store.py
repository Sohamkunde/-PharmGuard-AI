"""Dual Index Store for PharmGuard: ChromaDB Vector Store + BM25 Lexical Index.

1. Initializes a persistent ChromaDB client at ./chroma_db with collection 'pharmguard_clinical'.
2. Employs sentence-transformers with 'BAAI/bge-base-en-v1.5' for dense vector embeddings.
3. Inserts sentence-window chunks into ChromaDB (individual sentence as searchable document,
   expanded window in metadata).
4. Builds a sparse lexical index using rank_bm25 (BM25Okapi) and persists to ./chroma_db/bm25_index.pkl.
"""

import os
import re
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("store")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "corpus_chunks.jsonl"
CHROMA_DIR = BASE_DIR / "chroma_db"
BM25_INDEX_FILE = CHROMA_DIR / "bm25_index.pkl"

COLLECTION_NAME = "pharmguard_clinical"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 256


def tokenize_bm25(text: str) -> List[str]:
    """Tokenize clinical text for BM25 lexical indexing."""
    if not text:
        return []
    # Tokenize words, hyphens, and medical alphanumeric acronyms
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-\./]+\b", text.lower())
    clean_tokens = []
    for token in raw_tokens:
        t = token.strip(".-/")
        if len(t) > 1 and not t.isdigit():
            clean_tokens.append(t)
    return clean_tokens


def get_embedding_function(model_name: str = EMBEDDING_MODEL_NAME):
    """Initialize SentenceTransformer embedding function with normalized embeddings."""
    logger.info("Initializing SentenceTransformerEmbeddingFunction with '%s'...", model_name)
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        normalize_embeddings=True,
    )


def load_corpus_chunks(chunks_file: Path = CHUNKS_FILE) -> List[Dict[str, Any]]:
    """Load pre-processed sentence-window chunks from JSONL."""
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_file}. Run chunker.py first.")

    chunks = []
    logger.info("Loading corpus chunks from %s...", chunks_file)
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    logger.info("Loaded %d chunks from %s", len(chunks), chunks_file)
    return chunks


def index_in_chromadb(
    chunks: List[Dict[str, Any]],
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_func = None,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Insert sentence-window chunks into persistent ChromaDB collection."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing persistent ChromaDB client at %s...", chroma_dir)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if embedding_func is None:
        embedding_func = get_embedding_function()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = len(chunks)
    logger.info("Upserting %d chunks into ChromaDB in batches of %d...", total_chunks, batch_size)

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        ids = [c["chunk_id"] for c in batch]
        documents = [c["sentence"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(
            "ChromaDB batch progress: %d/%d (%.1f%%)",
            min(i + batch_size, total_chunks),
            total_chunks,
            (min(i + batch_size, total_chunks) / total_chunks) * 100,
        )

    count = collection.count()
    logger.info("ChromaDB indexing complete. Total collection count: %d", count)
    return count


def build_and_save_bm25_index(
    chunks: List[Dict[str, Any]],
    bm25_output_file: Path = BM25_INDEX_FILE,
) -> int:
    """Build BM25Okapi sparse lexical index and save to pickle."""
    bm25_output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building BM25Okapi index over %d documents...", len(chunks))

    chunk_ids = [c["chunk_id"] for c in chunks]
    sentences = [c["sentence"] for c in chunks]
    windows = [c["window"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    logger.info("Tokenizing corpus for BM25...")
    tokenized_corpus = [tokenize_bm25(s) for s in sentences]

    bm25 = BM25Okapi(tokenized_corpus)

    payload = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "sentences": sentences,
        "windows": windows,
        "metadatas": metadatas,
        "corpus_size": len(chunks),
    }

    logger.info("Saving BM25 index payload to %s...", bm25_output_file)
    with open(bm25_output_file, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("BM25 index successfully built and saved. Corpus size: %d", len(chunks))
    return len(chunks)


def verify_indices(chroma_dir: Path = CHROMA_DIR, bm25_file: Path = BM25_INDEX_FILE) -> Dict[str, Any]:
    """Verify that both indices load correctly and answer a sample clinical query."""
    logger.info("Verifying ChromaDB and BM25 indices...")
    results = {}

    # 1. Verify ChromaDB
    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_func = get_embedding_function()
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)
    chroma_count = collection.count()
    results["chromadb_count"] = chroma_count
    logger.info("ChromaDB verified: %d chunks found.", chroma_count)

    # 2. Verify BM25
    with open(bm25_file, "rb") as f:
        bm25_data = pickle.load(f)
    bm25_count = bm25_data["corpus_size"]
    results["bm25_count"] = bm25_count
    logger.info("BM25 index verified: %d documents in corpus.", bm25_count)

    # 3. Test Query Execution
    test_query = "Paxlovid amiodarone drug interaction contraindication"
    
    # Dense test
    dense_res = collection.query(query_texts=[test_query], n_results=2)
    top_dense_id = dense_res["ids"][0][0] if dense_res["ids"] and dense_res["ids"][0] else None
    top_dense_meta = dense_res["metadatas"][0][0] if dense_res["metadatas"] and dense_res["metadatas"][0] else {}
    results["dense_test_top_id"] = top_dense_id
    results["dense_test_drug"] = top_dense_meta.get("drug_name")
    logger.info("Test dense query match: %s (Drug: %s)", top_dense_id, results["dense_test_drug"])

    # Sparse BM25 test
    q_tokens = tokenize_bm25(test_query)
    scores = bm25_data["bm25"].get_scores(q_tokens)
    top_idx = scores.argmax() if len(scores) > 0 else -1
    if top_idx >= 0:
        results["sparse_test_top_id"] = bm25_data["chunk_ids"][top_idx]
        results["sparse_test_drug"] = bm25_data["metadatas"][top_idx].get("drug_name")
        results["sparse_test_score"] = float(scores[top_idx])
        logger.info(
            "Test sparse query match: %s (Drug: %s, Score: %.3f)",
            results["sparse_test_top_id"],
            results["sparse_test_drug"],
            results["sparse_test_score"],
        )

    return results


def run_indexing():
    """Execute complete indexing pipeline."""
    chunks = load_corpus_chunks()
    ef = get_embedding_function()
    chroma_count = index_in_chromadb(chunks=chunks, embedding_func=ef)
    bm25_count = build_and_save_bm25_index(chunks=chunks)
    verification = verify_indices()
    print("\n" + "=" * 50)
    print("PHASE 2 DUAL INDEXING VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Total Chunks Loaded from JSONL: {len(chunks)}")
    print(f"Total Chunks in ChromaDB:       {chroma_count}")
    print(f"Total Documents in BM25:        {bm25_count}")
    print(f"ChromaDB Collection Name:       {COLLECTION_NAME}")
    print(f"BM25 Index File:                {BM25_INDEX_FILE}")
    print(f"Dense Verification Top Hit:     {verification.get('dense_test_top_id')} ({verification.get('dense_test_drug')})")
    print(f"BM25 Verification Top Hit:      {verification.get('sparse_test_top_id')} ({verification.get('sparse_test_drug')})")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_indexing()
