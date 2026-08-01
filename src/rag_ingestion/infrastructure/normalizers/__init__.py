"""
Document normalizer implementations.

Normalizers transform the structured document representation produced by
Docling into a normalized, deterministic representation suitable for
chunking while preserving the logical structure of the source document.

Responsibilities:

- Normalize whitespace
- Normalize line endings
- Normalize character encoding artifacts
- Remove boilerplate where appropriate
- Preserve document hierarchy
- Preserve section boundaries
- Preserve page references
- Produce deterministic normalized output

Non-responsibilities:

- Loading source documents
- Parsing document formats
- Chunk generation
- Metadata enrichment
- Embedding generation
- Vector storage

Planned implementations:

- DefaultDocumentNormalizer

Normalization should be deterministic so that identical source documents
always produce identical normalized representations.
"""

from rag_ingestion.infrastructure.normalizers.default import DefaultDocumentNormalizer

__all__ = ["DefaultDocumentNormalizer"]