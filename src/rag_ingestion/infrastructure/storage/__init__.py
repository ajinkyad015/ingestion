"""
Vector store implementations.

This package contains infrastructure adapters responsible for persisting
embedded document chunks and their associated metadata.

Version 1 of the ingestion pipeline uses ChromaDB as the backing vector
database. The application layer communicates only through the
VectorStore protocol defined in the Domain layer, allowing additional
vector databases to be introduced without changing orchestration logic.

Current implementation:

- ChromaVectorStore

Responsibilities:

- Create or connect to a vector collection
- Persist embeddings
- Persist chunk text
- Persist deterministic metadata
- Persist document identifiers
- Support batch insertion
- Support retrieval by document identifier
- Support deletion by document identifier

Non-responsibilities:

- Document loading
- Parsing
- Normalization
- Chunk generation
- Embedding generation

Configuration is provided through the Settings object:

- CHROMA_PERSIST_DIRECTORY
- CHROMA_COLLECTION

The implementation uses Chroma's persistent storage backend so that
ingested documents remain available across application restarts.

Future vector store adapters may include:

- PostgreSQL + pgvector
- Qdrant
- Weaviate
- Milvus
- Pinecone

These implementations should satisfy the same VectorStore protocol,
allowing the application layer to remain unchanged.
"""

from rag_ingestion.infrastructure.storage.chroma import ChromaVectorStore

__all__ = ["ChromaVectorStore"]