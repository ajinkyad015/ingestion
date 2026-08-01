"""
Domain layer.

The domain layer contains the core business abstractions of the RAG ingestion
pipeline. It is independent of infrastructure concerns and defines the
entities and protocols that the application depends upon.

Dependency direction:

    Domain
        ↑
Application
        ↑
Infrastructure
        ↑
Interface

No code in this package may depend on the Application, Infrastructure,
or Interface layers.
"""

from rag_ingestion.domain.entities import (
    Chunk,
    ChunkMetadata,
    Document,
    EmbeddedChunk,
    NormalizedDocument,
    ParsedDocument,
)
from rag_ingestion.domain.protocols import (
    Chunker,
    Embedder,
    Loader,
    MetadataEnricher,
    Normalizer,
    Parser,
    VectorStore,
)

__all__ = [
    # Entities
    "Chunk",
    "ChunkMetadata",
    "Document",
    "EmbeddedChunk",
    "NormalizedDocument",
    "ParsedDocument",
    # Protocols
    "Chunker",
    "Embedder",
    "Loader",
    "MetadataEnricher",
    "Normalizer",
    "Parser",
    "VectorStore",
]