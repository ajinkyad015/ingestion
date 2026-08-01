from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_ingestion.domain.entities.chunk import Chunk
from rag_ingestion.domain.entities.document import NormalizedDocument


@runtime_checkable
class Chunker(Protocol):
    """
    Contract for splitting normalized documents into retrieval-optimized chunks.

    A Chunker receives a ``NormalizedDocument`` and returns an ordered list of
    ``Chunk`` objects whose IDs are deterministic and whose metadata preserves
    provenance information.

    Chunk ordering is always the same for identical inputs.
    """

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        """
        Split *document* into an ordered list of chunks.

        Parameters
        ----------
        document:
            A normalized document ready for chunking.

        Returns
        -------
        list[Chunk]
            Ordered list of chunks.  Returns a list with a single chunk
            when the normalized text is shorter than the configured chunk
            size.  Returns an empty list when the normalized text is empty.
        """
        ...
