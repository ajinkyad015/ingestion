from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    """
    Represents a raw source document before any processing.

    Attributes
    ----------
    source_path:
        Absolute path to the document on the filesystem.
    extension:
        Lowercased file extension including the leading dot (e.g. ``.pdf``).
    source_hash:
        SHA-256 hex digest of the raw file bytes, used for deduplication.
    size_bytes:
        Total byte size of the source file.
    """

    source_path: Path
    extension: str
    source_hash: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """
    Represents a document after Docling has extracted its content.

    Attributes
    ----------
    source:
        The originating raw document.
    text:
        Full extracted text content, with section structure preserved via
        double newlines between sections.
    title:
        Document title if present in the source; empty string otherwise.
    num_pages:
        Total page count for paginated formats (e.g. PDF). Zero for
        unpaginated formats (e.g. plain text, Markdown).
    """

    source: Document
    text: str
    title: str
    num_pages: int


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    """
    Represents a parsed document after whitespace and encoding normalization.

    The normalized form is deterministic: identical parsed inputs always
    produce identical normalized outputs.

    Attributes
    ----------
    source:
        The parsed document this was produced from.
    text:
        Normalized text content ready for chunking.
    sections:
        Ordered list of (heading, body) pairs extracted from the document
        structure.  Heading is an empty string for preamble text that
        precedes the first heading.
    """

    source: ParsedDocument
    text: str
    sections: tuple[tuple[str, str], ...] = field(default_factory=tuple)
