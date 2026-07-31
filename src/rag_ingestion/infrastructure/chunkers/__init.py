"""
Chunking implementations.

Chunkers are responsible for transforming normalized document content into
retrieval-optimized chunks while preserving enough contextual information
to support downstream embedding and retrieval.

Version 1 of the ingestion pipeline intentionally supports a single,
configurable chunking strategy to keep behavior deterministic and easy to
evaluate.

Current strategy:

- Recursive chunking

Responsibilities:

- Split normalized content into chunks
- Respect configured chunk size
- Respect configured chunk overlap
- Preserve section boundaries where possible
- Produce deterministic chunk ordering

Non-responsibilities:

- Loading documents
- Parsing document formats
- Normalization
- Metadata enrichment
- Embedding generation
- Vector storage

Planned implementations:

- RecursiveChunker

Chunk size and overlap are configured through the application Settings
class rather than hardcoded values.
"""

__all__: list[str] = []