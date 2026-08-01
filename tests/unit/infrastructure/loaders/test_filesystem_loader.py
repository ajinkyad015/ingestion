"""
Unit tests for FileSystemLoader.

All tests are fully isolated: no real filesystem access occurs.
pytest-mock (``mocker``) is used to patch ``pathlib.Path`` methods so that
the loader behaves deterministically regardless of the host environment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_ingestion.config import Settings
from rag_ingestion.domain.entities.document import Document
from rag_ingestion.domain.protocols.loader import Loader
from rag_ingestion.infrastructure.loaders.filesystem import FileSystemLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(
    supported_extensions: str = ".pdf,.docx,.txt,.md,.markdown,.html,.htm",
    hash_algorithm: str = "sha256",
) -> Settings:
    """Return a Settings instance with controlled field values."""
    return Settings(
        SUPPORTED_EXTENSIONS=supported_extensions,
        HASH_ALGORITHM=hash_algorithm,
    )


def _fake_file(
    path: str,
    content: bytes = b"hello",
) -> MagicMock:
    """
    Build a ``MagicMock`` that behaves like a ``pathlib.Path`` file object.
    """
    mock = MagicMock(spec=Path)
    mock.is_file.return_value = True
    mock.is_dir.return_value = False
    mock.suffix = Path(path).suffix
    mock.read_bytes.return_value = content
    mock.resolve.return_value = Path(path)
    mock.__str__ = lambda self: path  # type: ignore[assignment]
    return mock


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestFileSystemLoaderProtocol:
    def test_satisfies_loader_protocol(self) -> None:
        settings = _make_settings()
        loader = FileSystemLoader(settings)
        assert isinstance(loader, Loader)


# ---------------------------------------------------------------------------
# Constructor / configuration
# ---------------------------------------------------------------------------


class TestFileSystemLoaderInit:
    def test_supported_extensions_stored(self) -> None:
        settings = _make_settings(supported_extensions=".pdf,.txt")
        loader = FileSystemLoader(settings)
        assert loader._supported_extensions == (".pdf", ".txt")

    def test_hash_algorithm_stored(self) -> None:
        settings = _make_settings(hash_algorithm="sha256")
        loader = FileSystemLoader(settings)
        assert loader._hash_algorithm == "sha256"

    def test_custom_extension_set(self) -> None:
        settings = _make_settings(supported_extensions=".md,.markdown")
        loader = FileSystemLoader(settings)
        assert ".md" in loader._supported_extensions
        assert ".markdown" in loader._supported_extensions
        assert ".pdf" not in loader._supported_extensions


# ---------------------------------------------------------------------------
# load() — input validation
# ---------------------------------------------------------------------------


class TestFileSystemLoaderLoadValidation:
    def test_raises_value_error_for_nonexistent_path(self) -> None:
        settings = _make_settings()
        loader = FileSystemLoader(settings)
        with patch.object(Path, "is_dir", return_value=False):
            with pytest.raises(ValueError, match="not an existing directory"):
                loader.load("/nonexistent/path")

    def test_raises_value_error_for_file_path(self, tmp_path: Path) -> None:
        """Passing a file (not a directory) must raise ValueError."""
        settings = _make_settings()
        loader = FileSystemLoader(settings)
        # tmp_path is a real dir; we make a file inside, then pass IT as source
        real_file = tmp_path / "doc.txt"
        real_file.write_bytes(b"content")
        with pytest.raises(ValueError, match="not an existing directory"):
            loader.load(str(real_file))

    def test_error_message_contains_source(self) -> None:
        settings = _make_settings()
        loader = FileSystemLoader(settings)
        bad_path = "/absolutely/not/here"
        with patch.object(Path, "is_dir", return_value=False):
            with pytest.raises(ValueError, match=bad_path):
                loader.load(bad_path)


# ---------------------------------------------------------------------------
# load() — happy paths (filesystem mocked)
# ---------------------------------------------------------------------------


class TestFileSystemLoaderLoad:
    def test_returns_empty_list_when_no_files(self) -> None:
        settings = _make_settings()
        loader = FileSystemLoader(settings)
        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "rglob", return_value=iter([])),
        ):
            result = loader.load("/some/dir")
        assert result == []

    def test_skips_unsupported_extensions(self) -> None:
        settings = _make_settings(supported_extensions=".pdf")
        loader = FileSystemLoader(settings)

        unsupported = _fake_file("/dir/image.png", b"png-data")
        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "rglob", return_value=iter([unsupported])),
        ):
            result = loader.load("/dir")
        assert result == []

    def test_returns_document_for_supported_extension(self) -> None:
        settings = _make_settings(supported_extensions=".txt")
        loader = FileSystemLoader(settings)
        content = b"hello world"
        file_mock = _fake_file("/dir/readme.txt", content)

        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "rglob", return_value=iter([file_mock])),
        ):
            result = loader.load("/dir")

        assert len(result) == 1
        doc = result[0]
        assert isinstance(doc, Document)
        assert doc.extension == ".txt"
        assert doc.size_bytes == len(content)

    def test_document_hash_is_sha256_of_content(self) -> None:
        settings = _make_settings(supported_extensions=".txt", hash_algorithm="sha256")
        loader = FileSystemLoader(settings)
        content = b"deterministic content"
        expected_hash = hashlib.sha256(content).hexdigest()
        file_mock = _fake_file("/dir/file.txt", content)

        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "rglob", return_value=iter([file_mock])),
        ):
            result = loader.load("/dir")

        assert result[0].source_hash == expected_hash

    def test_skips_subdirectories_in_rglob_results(self) -> None:
        settings = _make_settings(supported_extensions=".pdf")
        loader = FileSystemLoader(settings)
        dir_mock = MagicMock(spec=Path)
        dir_mock.is_file.return_value = False

        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "rglob", return_value=iter([dir_mock])),
        ):
            result = loader.load("/dir")
        assert result == []

    def test_multiple_supported_files_returned(self) -> None:
        settings = _make_settings(supported_extensions=".pdf,.txt")
        loader = FileSystemLoader(settings)
        pdf = _fake_file("/dir/a.pdf", b"pdf bytes")
        txt = _fake_file("/dir/b.txt", b"txt bytes")

        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "rglob", return_value=iter([pdf, txt])),
        ):
            result = loader.load("/dir")

        assert len(result) == 2
        extensions = {doc.extension for doc in result}
        assert extensions == {".pdf", ".txt"}

    def test_mixed_supported_and_unsupported_filters_correctly(self) -> None:
        settings = _make_settings(supported_extensions=".md")
        loader = FileSystemLoader(settings)
        md_file = _fake_file("/dir/notes.md", b"# Title")
        png_file = _fake_file("/dir/photo.png", b"\x89PNG")
        csv_file = _fake_file("/dir/data.csv", b"a,b,c")

        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "rglob", return_value=iter([md_file, png_file, csv_file])),
        ):
            result = loader.load("/dir")

        assert len(result) == 1
        assert result[0].extension == ".md"

    def test_extension_matching_is_case_insensitive(self) -> None:
        """
        Files with uppercase extensions (e.g. ``.PDF``) should be matched
        because the loader normalises extensions to lowercase before comparing.
        """
        settings = _make_settings(supported_extensions=".pdf")
        loader = FileSystemLoader(settings)
        upper_pdf = _fake_file("/dir/DOC.PDF", b"pdf content")
        # Override suffix to simulate uppercase extension from the OS
        upper_pdf.suffix = ".PDF"

        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "rglob", return_value=iter([upper_pdf])),
        ):
            result = loader.load("/dir")

        assert len(result) == 1
        assert result[0].extension == ".pdf"

    def test_size_bytes_matches_content_length(self) -> None:
        settings = _make_settings(supported_extensions=".docx")
        loader = FileSystemLoader(settings)
        content = b"A" * 1024
        file_mock = _fake_file("/dir/report.docx", content)

        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "rglob", return_value=iter([file_mock])),
        ):
            result = loader.load("/dir")

        assert result[0].size_bytes == 1024

    def test_empty_file_produces_valid_document(self) -> None:
        settings = _make_settings(supported_extensions=".txt")
        loader = FileSystemLoader(settings)
        file_mock = _fake_file("/dir/empty.txt", b"")

        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "rglob", return_value=iter([file_mock])),
        ):
            result = loader.load("/dir")

        doc = result[0]
        assert doc.size_bytes == 0
        assert doc.source_hash == hashlib.sha256(b"").hexdigest()

    def test_documents_are_frozen(self) -> None:
        """Document instances returned must be immutable (frozen dataclass)."""
        settings = _make_settings(supported_extensions=".txt")
        loader = FileSystemLoader(settings)
        file_mock = _fake_file("/dir/file.txt", b"data")

        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "rglob", return_value=iter([file_mock])),
        ):
            result = loader.load("/dir")

        with pytest.raises((AttributeError, TypeError)):
            result[0].extension = ".pdf"  # type: ignore[misc]
