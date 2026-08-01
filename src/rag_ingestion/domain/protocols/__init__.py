"""
Domain protocol definitions.

This package contains the abstract interfaces (Protocols) that define the
contracts for each stage of the ingestion pipeline.

Every infrastructure implementation must satisfy one of these protocols.
The application layer depends exclusively on these abstractions, allowing
infrastructure implementations to be substituted without modifying
orchestrators or business logic.

Protocols:

- ``Loader``      — discover and load source documents.
- ``Parser``      — extract structured text from a raw document.
- ``Normalizer``  — clean and canonicalize parsed text.
- ``Chunker``     — split normalized text into retrieval chunks.
- ``MetadataEnricher`` — enrich chunk metadata before embedding.
- ``Embedder``    — encode chunks as dense vectors.
- ``VectorStore`` — persist and retrieve embedded chunks.
"""

from rag_ingestion.domain.protocols.chunker import Chunker
from rag_ingestion.domain.protocols.embedder import Embedder
from rag_ingestion.domain.protocols.loader import Loader
from rag_ingestion.domain.protocols.metadata_enricher import MetadataEnricher
from rag_ingestion.domain.protocols.normalizer import Normalizer
from rag_ingestion.domain.protocols.parser import Parser
from rag_ingestion.domain.protocols.vector_store import VectorStore

__all__ = [
    "Chunker",
    "Embedder",
    "Loader",
    "MetadataEnricher",
    "Normalizer",
    "Parser",
    "VectorStore",
]