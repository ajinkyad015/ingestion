from __future__ import annotations

from pathlib import Path

import pytest
from rag_ingestion.domain.entities.document import Document, ParsedDocument
from rag_ingestion.domain.protocols.normalizer import Normalizer
from rag_ingestion.infrastructure.normalizers.default import DefaultDocumentNormalizer


@pytest.fixture
def dummy_document() -> Document:
    return Document(
        source_path=Path("/docs/test.txt"),
        extension=".txt",
        source_hash="hash",
        size_bytes=100,
    )


@pytest.fixture
def dummy_parsed(dummy_document: Document) -> ParsedDocument:
    return ParsedDocument(
        source=dummy_document,
        text="",
        title="Test",
        num_pages=1,
    )


class TestDefaultDocumentNormalizer:
    def test_satisfies_protocol(self) -> None:
        normalizer = DefaultDocumentNormalizer()
        assert isinstance(normalizer, Normalizer)

    def test_normalize_whitespace(self, dummy_parsed: ParsedDocument) -> None:
        text = "Line 1 \t \nLine 2  \nLine 3"
        dummy = ParsedDocument(source=dummy_parsed.source, text=text, title="", num_pages=0)
        normalizer = DefaultDocumentNormalizer()
        
        result = normalizer.normalize(dummy)
        
        assert result.text == "Line 1\nLine 2\nLine 3"

    def test_normalize_unicode(self, dummy_parsed: ParsedDocument) -> None:
        # NFD form of "é"
        nfd_text = "e\u0301"
        dummy = ParsedDocument(source=dummy_parsed.source, text=nfd_text, title="", num_pages=0)
        normalizer = DefaultDocumentNormalizer()
        
        result = normalizer.normalize(dummy)
        
        # NFC form of "é"
        assert result.text == "\u00e9"

    def test_normalize_repeated_blank_lines(self, dummy_parsed: ParsedDocument) -> None:
        text = "Para 1\n\n\n\nPara 2\n\n\nPara 3"
        dummy = ParsedDocument(source=dummy_parsed.source, text=text, title="", num_pages=0)
        normalizer = DefaultDocumentNormalizer()
        
        result = normalizer.normalize(dummy)
        
        assert result.text == "Para 1\n\nPara 2\n\nPara 3"

    def test_section_preservation(self, dummy_parsed: ParsedDocument) -> None:
        text = "Preamble text\n\n# Heading 1\nContent 1\n\n## Heading 2\nContent 2"
        dummy = ParsedDocument(source=dummy_parsed.source, text=text, title="", num_pages=0)
        normalizer = DefaultDocumentNormalizer()
        
        result = normalizer.normalize(dummy)
        
        expected_sections = (
            ("", "Preamble text"),
            ("Heading 1", "Content 1"),
            ("Heading 2", "Content 2"),
        )
        assert result.sections == expected_sections

    def test_empty_document_handling(self, dummy_parsed: ParsedDocument) -> None:
        dummy = ParsedDocument(source=dummy_parsed.source, text="   \n   \n", title="", num_pages=0)
        normalizer = DefaultDocumentNormalizer()
        
        result = normalizer.normalize(dummy)
        
        assert result.text == ""
        assert result.sections == ()

    def test_deterministic_output(self, dummy_parsed: ParsedDocument) -> None:
        text = "Same text \r\n  with \xa0 variations\n\n\n"
        dummy = ParsedDocument(source=dummy_parsed.source, text=text, title="", num_pages=0)
        normalizer = DefaultDocumentNormalizer()
        
        result1 = normalizer.normalize(dummy)
        result2 = normalizer.normalize(dummy)
        
        assert result1 == result2

    def test_line_endings(self, dummy_parsed: ParsedDocument) -> None:
        text = "Line 1\r\nLine 2\rLine 3\n"
        dummy = ParsedDocument(source=dummy_parsed.source, text=text, title="", num_pages=0)
        normalizer = DefaultDocumentNormalizer()
        
        result = normalizer.normalize(dummy)
        
        assert result.text == "Line 1\nLine 2\nLine 3"
