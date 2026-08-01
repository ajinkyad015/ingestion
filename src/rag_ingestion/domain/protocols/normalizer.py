from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_ingestion.domain.entities.document import NormalizedDocument, ParsedDocument


@runtime_checkable
class Normalizer(Protocol):
    """
    Contract for text normalization of parsed documents.

    A Normalizer transforms a ``ParsedDocument`` into a ``NormalizedDocument``
    by applying deterministic text cleaning: whitespace normalization, line-
    ending canonicalization, and boilerplate removal.

    Normalization must be deterministic: identical inputs always produce
    identical outputs.
    """

    def normalize(self, document: ParsedDocument) -> NormalizedDocument:
        """
        Normalize *document* into a ``NormalizedDocument``.

        Parameters
        ----------
        document:
            A parsed document whose text will be normalized.

        Returns
        -------
        NormalizedDocument
            Cleaned document ready for chunking.
        """
        ...
