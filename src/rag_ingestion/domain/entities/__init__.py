"""
Domain entities.

Entities represent the core business objects that flow through the
RAG ingestion pipeline:

- ``Document`` — raw source file before any processing.
- ``ParsedDocument`` — content extracted by the parser.
- ``NormalizedDocument`` — cleaned, deterministic form ready for chunking.
- ``Chunk`` — a single retrieval-optimized text segment.
- ``ChunkMetadata`` — provenance data attached to every chunk.
- ``EmbeddedChunk`` — a chunk paired with its dense vector representation.

Entities are immutable (frozen dataclasses) and contain no infrastructure-
specific logic.
"""

from rag_ingestion.domain.entities.chunk import Chunk, ChunkMetadata, EmbeddedChunk
from rag_ingestion.domain.entities.document import Document, NormalizedDocument, ParsedDocument

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "Document",
    "EmbeddedChunk",
    "NormalizedDocument",
    "ParsedDocument",
]