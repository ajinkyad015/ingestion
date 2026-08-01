from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_ingestion.domain.entities.document import Document, ParsedDocument


@runtime_checkable
class Parser(Protocol):
    """
    Contract for converting raw documents into structured parsed form.

    A Parser receives a ``Document`` and returns a ``ParsedDocument`` whose
    ``text`` field contains the full extracted textual content.

    The concrete implementation delegates to Docling; this protocol keeps
    the application layer free of that dependency.
    """

    def parse(self, document: Document) -> ParsedDocument:
        """
        Parse *document* into a structured ``ParsedDocument``.

        Parameters
        ----------
        document:
            A loaded source document.

        Returns
        -------
        ParsedDocument
            Extracted text and metadata.

        Raises
        ------
        ValueError
            If *document* is of an unsupported format.
        RuntimeError
            If parsing fails for an unrecoverable reason.
        """
        ...
