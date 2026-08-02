"""
Unit tests for DoclingParser.

All tests are fully isolated:
- No real filesystem access.
- No network access.
- Docling is mocked completely; the real library is never invoked.

Test classes
------------
TestDoclingParserProtocol       — isinstance(parser, Parser)
TestDoclingParserInit           — constructor stores settings correctly
TestDoclingParserParse          — happy-path conversion to ParsedDocument
TestDoclingParserMetadata       — metadata (title, num_pages) preservation
TestDoclingParserErrorHandling  — RuntimeError on Docling failure
TestDoclingParserUnsupported    — ValueError for unsupported / malformed input
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rag_ingestion.config import Settings
from rag_ingestion.domain.entities.document import Document, ParsedDocument
from rag_ingestion.domain.protocols.parser import Parser
from rag_ingestion.infrastructure.parsers.docling_parser import DoclingParser

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_settings(enable_ocr: bool = False) -> Settings:
    """Return a Settings instance with controlled field values."""
    return Settings(DOCLING_ENABLE_OCR=enable_ocr)


def _make_document(
    path: str = "/docs/report.pdf",
    extension: str = ".pdf",
) -> Document:
    """Build a minimal Document for use in tests."""
    return Document(
        source_path=Path(path),
        extension=extension,
        source_hash="abc123",
        size_bytes=1024,
    )


def _make_converter_result(
    markdown_text: str = "# Hello\n\nWorld",
    title: str = "Test Title",
    num_pages: int = 3,
) -> MagicMock:
    """
    Build a MagicMock that looks like a Docling ConversionResult.

    Structure mirrored from Docling's public API:
    ``result.document.export_to_markdown()`` → str
    ``result.document.pages``               → sequence of page objects
    ``result.document.metadata.title``      → str
    """
    page_mocks = [MagicMock() for _ in range(num_pages)]

    meta_mock = MagicMock()
    meta_mock.title = title

    doc_mock = MagicMock()
    doc_mock.export_to_markdown.return_value = markdown_text
    doc_mock.pages = page_mocks
    doc_mock.meta = meta_mock
    doc_mock.name = ""

    result_mock = MagicMock()
    result_mock.document = doc_mock

    return result_mock


def _patch_converter(result: MagicMock) -> MagicMock:
    """
    Return a MagicMock ``DocumentConverter`` whose ``.convert()`` returns
    *result*.
    """
    converter = MagicMock()
    converter.convert.return_value = result
    return converter


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestDoclingParserProtocol:
    def test_satisfies_parser_protocol(self) -> None:
        """DoclingParser must be recognised as a Parser at runtime."""
        settings = _make_settings()
        with patch.object(DoclingParser, "_build_converter", return_value=MagicMock()):
            parser = DoclingParser(settings)
        assert isinstance(parser, Parser)


# ---------------------------------------------------------------------------
# Constructor / configuration
# ---------------------------------------------------------------------------


class TestDoclingParserInit:
    def test_ocr_flag_stored_false(self) -> None:
        settings = _make_settings(enable_ocr=False)
        with patch.object(DoclingParser, "_build_converter", return_value=MagicMock()):
            parser = DoclingParser(settings)
        assert parser._enable_ocr is False

    def test_ocr_flag_stored_true(self) -> None:
        settings = _make_settings(enable_ocr=True)
        with patch.object(DoclingParser, "_build_converter", return_value=MagicMock()):
            parser = DoclingParser(settings)
        assert parser._enable_ocr is True

    def test_build_converter_called_on_init(self) -> None:
        """_build_converter should be called exactly once during __init__."""
        settings = _make_settings()
        fake_converter = MagicMock()
        with patch.object(
            DoclingParser, "_build_converter", return_value=fake_converter
        ) as mock_build:
            parser = DoclingParser(settings)
        mock_build.assert_called_once_with(False)
        assert parser._converter is fake_converter


# ---------------------------------------------------------------------------
# parse() — happy paths
# ---------------------------------------------------------------------------


class TestDoclingParserParse:
    def test_returns_parsed_document(self) -> None:
        settings = _make_settings()
        document = _make_document()
        result = _make_converter_result()
        converter = _patch_converter(result)

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        parsed = parser.parse(document)

        assert isinstance(parsed, ParsedDocument)

    def test_parsed_document_source_is_original_document(self) -> None:
        settings = _make_settings()
        document = _make_document()
        result = _make_converter_result()
        converter = _patch_converter(result)

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        parsed = parser.parse(document)

        assert parsed.source is document

    def test_converter_called_with_source_path_string(self) -> None:
        settings = _make_settings()
        document = _make_document(path="/docs/report.pdf")
        result = _make_converter_result()
        converter = _patch_converter(result)

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        parser.parse(document)

        converter.convert.assert_called_once_with(str(document.source_path))

    def test_text_comes_from_export_to_markdown(self) -> None:
        settings = _make_settings()
        document = _make_document()
        expected_text = "# Section\n\nSome content here."
        result = _make_converter_result(markdown_text=expected_text)
        converter = _patch_converter(result)

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        parsed = parser.parse(document)

        assert parsed.text == expected_text

    @pytest.mark.parametrize("extension", [".pdf",
                                            ".docx", 
                                            ".txt", 
                                            ".md", 
                                            ".markdown", 
                                            ".html", 
                                            ".htm"
                                        ]
                            )
    def test_all_supported_extensions_parse_successfully(self, extension: str) -> None:
        settings = _make_settings()
        document = _make_document(path=f"/docs/file{extension}", extension=extension)
        result = _make_converter_result()
        converter = _patch_converter(result)

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        parsed = parser.parse(document)

        assert isinstance(parsed, ParsedDocument)

    def test_parsed_document_is_frozen(self) -> None:
        """ParsedDocument instances must be immutable (frozen dataclass)."""
        settings = _make_settings()
        document = _make_document()
        result = _make_converter_result()
        converter = _patch_converter(result)

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        parsed = parser.parse(document)

        with pytest.raises((AttributeError, TypeError)):
            parsed.text = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# parse() — metadata preservation
# ---------------------------------------------------------------------------


class TestDoclingParserMetadata:
    def test_title_preserved(self) -> None:
        settings = _make_settings()
        document = _make_document()
        result = _make_converter_result(title="Annual Report 2024")
        converter = _patch_converter(result)

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        parsed = parser.parse(document)

        assert parsed.title == "Annual Report 2024"

    def test_num_pages_preserved(self) -> None:
        settings = _make_settings()
        document = _make_document()
        result = _make_converter_result(num_pages=12)
        converter = _patch_converter(result)

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        parsed = parser.parse(document)

        assert parsed.num_pages == 12

    def test_empty_title_when_metadata_absent(self) -> None:
        settings = _make_settings()
        document = _make_document()

        # Result with no meta attribute at all
        doc_mock = MagicMock()
        doc_mock.export_to_markdown.return_value = "text"
        doc_mock.pages = []
        del doc_mock.meta           # ensure AttributeError on access
        del doc_mock.metadata
        doc_mock.name = ""

        result_mock = MagicMock()
        result_mock.document = doc_mock
        converter = _patch_converter(result_mock)

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        parsed = parser.parse(document)

        assert parsed.title == ""

    def test_zero_pages_for_unpaginated_formats(self) -> None:
        settings = _make_settings()
        document = _make_document(path="/docs/notes.txt", extension=".txt")

        doc_mock = MagicMock()
        doc_mock.export_to_markdown.return_value = "plain text"
        doc_mock.pages = []          # empty → num_pages == 0
        doc_mock.meta = MagicMock()
        doc_mock.meta.title = ""
        doc_mock.name = ""

        result_mock = MagicMock()
        result_mock.document = doc_mock
        converter = _patch_converter(result_mock)

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        parsed = parser.parse(document)

        assert parsed.num_pages == 0

    def test_title_falls_back_to_doc_name(self) -> None:
        """When meta.title is empty, parser falls back to doc_obj.name."""
        settings = _make_settings()
        document = _make_document()

        meta_mock = MagicMock()
        meta_mock.title = ""         # empty → trigger fallback

        doc_mock = MagicMock()
        doc_mock.export_to_markdown.return_value = "content"
        doc_mock.pages = [MagicMock()]
        doc_mock.meta = meta_mock
        doc_mock.name = "Fallback Name"

        result_mock = MagicMock()
        result_mock.document = doc_mock
        converter = _patch_converter(result_mock)

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        parsed = parser.parse(document)

        assert parsed.title == "Fallback Name"


# ---------------------------------------------------------------------------
# parse() — error handling
# ---------------------------------------------------------------------------


class TestDoclingParserErrorHandling:
    def test_runtime_error_on_converter_exception(self) -> None:
        settings = _make_settings()
        document = _make_document()
        converter = MagicMock()
        converter.convert.side_effect = Exception("internal docling error")

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        with pytest.raises(RuntimeError, match="failed to parse"):
            parser.parse(document)

    def test_runtime_error_contains_source_path(self) -> None:
        settings = _make_settings()
        document = _make_document(path="/data/broken.pdf")
        converter = MagicMock()
        converter.convert.side_effect = Exception("boom")

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        with pytest.raises(RuntimeError, match=r"broken\.pdf"):
            parser.parse(document)

    def test_runtime_error_chains_original_exception(self) -> None:
        settings = _make_settings()
        document = _make_document()
        original_exc = ValueError("docling internal")
        converter = MagicMock()
        converter.convert.side_effect = original_exc

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        with pytest.raises(RuntimeError) as exc_info:
            parser.parse(document)

        assert exc_info.value.__cause__ is original_exc

    def test_empty_text_when_export_raises(self) -> None:
        """If export_to_markdown() raises, text falls back to empty string."""
        settings = _make_settings()
        document = _make_document()

        doc_mock = MagicMock()
        doc_mock.export_to_markdown.side_effect = Exception("export error")
        doc_mock.pages = [MagicMock()]
        doc_mock.meta = MagicMock()
        doc_mock.meta.title = "Safe Title"
        doc_mock.name = ""

        result_mock = MagicMock()
        result_mock.document = doc_mock
        converter = _patch_converter(result_mock)

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        parsed = parser.parse(document)

        assert parsed.text == ""
        assert parsed.title == "Safe Title"


# ---------------------------------------------------------------------------
# parse() — unsupported / malformed input
# ---------------------------------------------------------------------------


class TestDoclingParserUnsupported:
    @pytest.mark.parametrize("extension", [".png", ".jpg", ".csv", ".xlsx", ".zip", ""])
    def test_value_error_for_unsupported_extension(self, extension: str) -> None:
        settings = _make_settings()
        document = _make_document(path=f"/docs/file{extension}", extension=extension)

        with patch.object(DoclingParser, "_build_converter", return_value=MagicMock()):
            parser = DoclingParser(settings)

        with pytest.raises(ValueError, match="does not support extension"):
            parser.parse(document)

    def test_value_error_contains_extension(self) -> None:
        settings = _make_settings()
        document = _make_document(path="/docs/image.png", extension=".png")

        with patch.object(DoclingParser, "_build_converter", return_value=MagicMock()):
            parser = DoclingParser(settings)

        with pytest.raises(ValueError, match=r"\.png"):
            parser.parse(document)

    def test_value_error_contains_source_path(self) -> None:
        settings = _make_settings()
        document = _make_document(path="/docs/image.png", extension=".png")

        with patch.object(DoclingParser, "_build_converter", return_value=MagicMock()):
            parser = DoclingParser(settings)

        with pytest.raises(ValueError, match="image.png"):
            parser.parse(document)

    def test_converter_not_called_for_unsupported_extension(self) -> None:
        settings = _make_settings()
        document = _make_document(path="/docs/data.csv", extension=".csv")
        converter = MagicMock()

        with patch.object(DoclingParser, "_build_converter", return_value=converter):
            parser = DoclingParser(settings)

        with pytest.raises(ValueError):
            parser.parse(document)

        converter.convert.assert_not_called()
