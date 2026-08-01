from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_ingestion.domain.entities.chunk import Chunk, EmbeddedChunk


@runtime_checkable
class Embedder(Protocol):
    """
    Contract for producing dense vector embeddings from text chunks.

    An Embedder takes a batch of ``Chunk`` objects and returns an equal-length
    list of ``EmbeddedChunk`` objects pairing each chunk with its embedding.

    Embeddings are produced deterministically for identical inputs when the
    underlying model is in inference mode.
    """

    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Embed *chunks* and return a paired list of ``EmbeddedChunk`` objects.

        Parameters
        ----------
        chunks:
            Non-empty list of chunks to embed.  Batching is handled
            internally according to the configured batch size.

        Returns
        -------
        list[EmbeddedChunk]
            Embedded chunks in the same order as the input.  The list
            length always equals ``len(chunks)``.

        Raises
        ------
        ValueError
            If *chunks* is empty.
        """
        ...
