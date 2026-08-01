from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """
    Provenance metadata attached to every chunk.

    Attributes
    ----------
    document_hash:
        SHA-256 hex digest of the originating source document bytes.
    source_path:
        Absolute path of the originating source document.
    chunk_index:
        Zero-based position of this chunk within its document.
    total_chunks:
        Total number of chunks produced from the originating document.
    heading:
        The most recent section heading that precedes this chunk.
        Empty string when no heading is present.
    page_number:
        Page number within the source document where this chunk begins.
        Zero for unpaginated formats.
    char_start:
        Character offset within the normalized document text where this
        chunk starts.
    char_end:
        Character offset within the normalized document text where this
        chunk ends (exclusive).
    """

    document_hash: str
    source_path: Path
    chunk_index: int
    total_chunks: int
    heading: str
    page_number: int
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class Chunk:
    """
    A single text segment produced by the chunking stage.

    Attributes
    ----------
    chunk_id:
        Deterministic identifier composed of the document hash and chunk
        index, formatted as ``{document_hash}_{chunk_index:06d}``.
    text:
        The raw text content of this chunk.
    metadata:
        Provenance and structural metadata for this chunk.
    """

    chunk_id: str
    text: str
    metadata: ChunkMetadata


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """
    A chunk paired with its dense vector embedding.

    Attributes
    ----------
    chunk:
        The originating text chunk.
    embedding:
        Dense vector representation produced by the embedding model.
        Length is determined by the configured embedding model.
    """

    chunk: Chunk
    embedding: tuple[float, ...]
