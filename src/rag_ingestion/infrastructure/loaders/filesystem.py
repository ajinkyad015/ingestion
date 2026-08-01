"""
Filesystem-based document loader.

Scans a directory for files whose extensions appear in the configured
allow-list, reads their bytes to compute a content hash, and returns
``Document`` instances.  No text extraction or parsing is performed here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from rag_ingestion.config import Settings
from rag_ingestion.domain.entities.document import Document


class FileSystemLoader:
    """
    Concrete ``Loader`` that discovers documents on the local filesystem.

    Parameters
    ----------
    settings:
        Application settings used to determine which file extensions are
        supported and which hash algorithm to apply.

    Raises
    ------
    ValueError
        Raised by :meth:`load` when *source* is not a directory.
    """

    def __init__(self, settings: Settings) -> None:
        self._supported_extensions: tuple[str, ...] = settings.supported_extensions_list
        self._hash_algorithm: str = settings.hash_algorithm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, source: str) -> list[Document]:
        """
        Discover supported documents under *source* directory.

        Walks the directory tree recursively and collects every file whose
        lowercased extension is present in ``supported_extensions_list``.
        Results are returned in a deterministic order (sorted by absolute
        path string).

        Parameters
        ----------
        source:
            Absolute or relative path to a directory on the local filesystem.

        Returns
        -------
        list[Document]
            ``Document`` instances for each discovered file, sorted by path.
            Returns an empty list when no supported files are found.

        Raises
        ------
        ValueError
            If *source* does not point to an existing directory.
        """
        directory = Path(source)
        if not directory.is_dir():
            raise ValueError(
                f"FileSystemLoader.load() requires a directory path; "
                f"got {source!r} which is not an existing directory."
            )

        documents: list[Document] = []
        for file_path in sorted(directory.rglob("*"), key=str):
            if not file_path.is_file():
                continue
            extension = file_path.suffix.lower()
            if extension not in self._supported_extensions:
                continue
            document = self._build_document(file_path)
            documents.append(document)

        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_document(self, file_path: Path) -> Document:
        """
        Read *file_path* and construct a :class:`Document`.

        Parameters
        ----------
        file_path:
            Resolved path to an existing, supported file.

        Returns
        -------
        Document
            Immutable domain entity describing the source file.
        """
        raw_bytes = file_path.read_bytes()
        source_hash = hashlib.new(self._hash_algorithm, raw_bytes).hexdigest()
        return Document(
            source_path=file_path.resolve(),
            extension=file_path.suffix.lower(),
            source_hash=source_hash,
            size_bytes=len(raw_bytes),
        )
