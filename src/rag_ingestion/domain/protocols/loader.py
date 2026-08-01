from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_ingestion.domain.entities.document import Document


@runtime_checkable
class Loader(Protocol):
    """
    Contract for document discovery and loading.

    A Loader is responsible for locating documents on a source (e.g. the
    filesystem), validating their extensions, and producing ``Document``
    instances that describe the source files.

    Loaders must NOT perform any parsing, text extraction, or transformation.
    """

    def load(self, source: str) -> list[Document]:
        """
        Discover and load documents from *source*.

        Parameters
        ----------
        source:
            An opaque locator understood by the concrete implementation.
            For the filesystem loader this is a directory path string.

        Returns
        -------
        list[Document]
            Ordered list of ``Document`` instances found at *source*.
            Returns an empty list when no supported documents are found.

        Raises
        ------
        ValueError
            If *source* is not a valid locator for this loader.
        """
        ...
