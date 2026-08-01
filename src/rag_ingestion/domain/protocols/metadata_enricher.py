from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_ingestion.domain.entities.chunk import Chunk


@runtime_checkable
class MetadataEnricher(Protocol):
    """
    Contract for enriching chunk metadata before embeddings are generated.

    A MetadataEnricher receives an ordered list of ``Chunk`` objects and
    returns an ordered list of ``Chunk`` objects whose metadata has been
    enriched deterministically.
    """

    def enrich(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Enrich *chunks* and return them in the same order.

        Parameters
        ----------
        chunks:
            Ordered list of chunks that need additional metadata.

        Returns
        -------
        list[Chunk]
            Enriched chunks in the same order as the input.
        """
        ...