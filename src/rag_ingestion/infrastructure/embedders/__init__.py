"""
Embedding adapter implementations.

This package contains infrastructure adapters responsible for converting
text chunks into dense vector representations for semantic search.

Version 1 of the ingestion pipeline uses Sentence Transformers from
Hugging Face. The application layer interacts only with the Embedder
Protocol defined in the domain layer, allowing alternative embedding
providers to be introduced without changing orchestration logic.

Current implementation:

- SentenceTransformerEmbedder

Responsibilities:

- Load the configured embedding model
- Encode document chunks into vectors
- Support configurable batch sizes
- Support configurable execution devices (CPU/GPU)
- Produce deterministic embeddings for identical inputs
- Optionally normalize embeddings

Non-responsibilities:

- Document loading
- Parsing
- Normalization
- Chunk generation
- Metadata enrichment
- Vector storage

Configuration is provided through the Settings object:

- EMBEDDING_MODEL
- EMBEDDING_BATCH_SIZE
- EMBEDDING_DEVICE
- EMBEDDING_NORMALIZE

Default model:

- BAAI/bge-small-en-v1.5

The embedding implementation should be replaceable without modifying
application services or orchestrators.
"""

__all__: list[str] = []