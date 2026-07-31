"""
Domain entities.

Entities represent the core business objects that flow through the
RAG ingestion pipeline.

Upcoming entities include:

- Document
- ParsedDocument
- NormalizedDocument
- Chunk
- ChunkMetadata
- EmbeddedChunk

Entities are immutable where practical and contain no infrastructure-
specific logic.
"""

__all__: list[str] = []