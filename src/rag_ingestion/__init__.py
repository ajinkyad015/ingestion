"""
RAG Ingestion

A production-grade Retrieval-Augmented Generation (RAG) ingestion pipeline
built using Clean Architecture principles.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rag-ingestion")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
]