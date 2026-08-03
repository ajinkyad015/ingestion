from pathlib import Path

from rag_ingestion.config import Settings
from rag_ingestion.infrastructure.parsers.docling_parser import DoclingParser
from rag_ingestion.domain.entities.document import Document

settings = Settings()

parser = DoclingParser(settings)

doc = Document(
    source_path=Path("path/to/doc.pdf"),
    extension=".pdf",
    source_hash="dummy",
    size_bytes=100,
)

parsed = parser.parse(doc)
print(parsed.text[:500])