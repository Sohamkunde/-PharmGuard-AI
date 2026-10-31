"""Retrieval modules: Hybrid BM25 + Dense ChromaDB retrieval with Cross-Encoder reranking."""

from src.agent.entity_resolver import DrugEntityResolver, get_entity_resolver

__all__ = ["DrugEntityResolver", "get_entity_resolver"]
