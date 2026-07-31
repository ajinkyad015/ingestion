"""
Document loader implementations.

Loaders are responsible only for locating, validating, and reading
document sources. They do not perform any parsing or document
transformation.

Responsibilities:

- Validate supported file types
- Resolve document paths
- Read document bytes
- Produce domain loader inputs

Non-responsibilities:

- Parsing document contents
- Text extraction
- OCR
- Metadata generation
- Chunking

Planned implementations:

- FileSystemLoader

Supported document types:

- TXT
- PDF
- DOCX
- Markdown
- HTML

All content parsing is delegated to the Docling parser adapter.
"""

__all__: list[str] = []