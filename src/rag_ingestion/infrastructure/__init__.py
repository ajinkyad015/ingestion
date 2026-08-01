"""
Infrastructure layer.

The infrastructure layer provides concrete implementations of the
interfaces defined in the domain layer.

This layer is responsible for integrating with external libraries,
frameworks, and services while keeping those dependencies isolated
from the rest of the application.

Current planned adapters:

- Filesystem document loaders
- Docling document parser
- Document normalizer
- Recursive chunker
- SentenceTransformer embedder
- ChromaDB vector store

Dependency rules:

    Domain
        ↑
    Application
        ↑
    Infrastructure

Infrastructure may depend on Domain and Application abstractions, but
must never be depended upon directly by the Domain layer.
"""

__all__: list[str] = []