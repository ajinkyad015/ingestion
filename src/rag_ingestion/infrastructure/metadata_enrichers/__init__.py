"""
Metadata enricher implementations.

This package contains infrastructure adapters responsible for attaching
deterministic provenance metadata to chunks before embeddings are generated.
"""

from rag_ingestion.infrastructure.metadata_enrichers.default import DefaultMetadataEnricher

__all__ = [
    "DefaultMetadataEnricher",
]