"""
Document parser implementations.

This package contains infrastructure adapters responsible for converting
supported document formats into a unified document representation using
Docling.

The parser layer delegates all format-specific parsing to Docling and does
not implement custom parsing logic for individual file types.

Responsibilities:

- Invoke Docling for document conversion
- Produce a structured document model
- Preserve document hierarchy
- Preserve headings
- Preserve page numbers
- Preserve tables
- Preserve captions
- Preserve lists
- Preserve layout metadata
- Surface parsing failures with consistent exceptions

Non-responsibilities:

- Document discovery
- Chunking
- Text normalization
- Metadata enrichment
- Embedding generation
- Vector storage

Planned implementations:

- DoclingParser

Supported formats (via Docling):

- PDF
- DOCX
- TXT
- Markdown
- HTML

Future support automatically benefits from Docling as additional document
formats become available without requiring new parser implementations.
"""

from rag_ingestion.infrastructure.parsers.docling_parser import DoclingParser

__all__: list[str] = ["DoclingParser"]