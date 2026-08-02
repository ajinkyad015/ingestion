# Repository Tree — `rag-ingestion`

> **Purpose of this file:** Give any agent an instant, token-efficient snapshot of the
> current repo state. Read this file first; skip full directory traversal.
>
> **Maintainance rule:** Update this file in every commit that adds, removes, or
> meaningfully changes a file.

---

## Last updated

**Commit 7 — "Implement DefaultMetadataEnricher adapter"**  
Local time: 2026-08-01 · Python ≥ 3.12.13 · uv · hatchling

---

## Stack (locked)

| Concern        | Library                          |
|----------------|----------------------------------|
| Parsing        | Docling ≥ 2.40                   |
| Embeddings     | sentence-transformers ≥ 3.0      |
| Default model  | `BAAI/bge-small-en-v1.5`         |
| Vector DB      | ChromaDB ≥ 1.0                   |
| Validation     | Pydantic v2                      |
| Configuration  | pydantic-settings                |
| Logging        | structlog                        |
| Testing        | pytest + pytest-mock             |
| Linting        | ruff                             |
| Type checking  | mypy (strict)                    |

---

## Architecture

```
Domain  →  Application  →  Infrastructure  →  Interface
```

Dependencies always point **inward**. Infrastructure may depend on Domain.
The Interface layer depends on Application + Domain only.
Constructor injection throughout; single composition root (`bootstrap.py`).

---

## Current progress

- Core scaffolding exists: `config.py`, `logging.py`, `bootstrap.py`, the CLI entrypoint, the domain entities/protocols, the metadata enricher protocol and adapter, the filesystem loader, the Docling parser adapter, and the default document normalizer.
- Implemented dependency chain so far: `Settings` → `FileSystemLoader` → `Document` → `DoclingParser` → `ParsedDocument` → `DefaultDocumentNormalizer` → `NormalizedDocument` → `DefaultMetadataEnricher`; `bootstrap()` resolves settings, configures logging, and returns `ApplicationContext`.
- `interface/cli/main.py` already depends on `bootstrap()` and can print the active configuration with `--config`.
- Application services and orchestrators are still stubs, so future ingestion features still need  embedder, and vector store adapters before they can be wired end to end.
- Existing tests currently anchor config, bootstrap, logging, domain entities/protocols, filesystem loader behavior, and the Docling parser adapter.
- For new features, keep the dependency order in mind: Domain stays dependency-free, Infrastructure plugs into Domain protocols, Application orchestrates, and Interface only composes through `bootstrap.py`.

---

## Root

```
d:\ingestion\
├── .env.example          # All env-var keys with documented defaults
├── .gitignore
├── .python-version       # 3.12.x pin for uv
├── pyproject.toml        # Build config, deps, pytest/ruff/mypy settings
├── uv.lock               # Locked dependency graph (181 packages)
├── README.md             # Project overview
├── README.pdf            # PDF render of README
└── REPO_TREE.md          # ← THIS FILE
```

---

## `src/rag_ingestion/` — main package

```
src/rag_ingestion/
├── __init__.py           # Exposes __version__ (importlib.metadata)
├── config.py             # Settings (pydantic-settings BaseSettings)
│                           Fields: APP_NAME, APP_ENV, APP_DEBUG,
│                                   LOG_LEVEL, LOG_FORMAT,
│                                   SUPPORTED_EXTENSIONS,
│                                   CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_STRATEGY,
│                                   EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
│                                   EMBEDDING_DEVICE, EMBEDDING_NORMALIZE,
│                                   CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION,
│                                   HASH_ALGORITHM, DOCLING_ENABLE_OCR,
│                                   MAX_WORKERS, DEFAULT_INPUT_DIRECTORY
│                           Helper: supported_extensions_list → tuple[str,...]
│                           Factory: get_settings() → Settings  (lru_cache)
├── logging.py            # configure_logging(settings) + get_logger(**ctx)
│                           Uses structlog; json or console renderer
└── bootstrap.py          # Single composition root
                            ApplicationContext(frozen dataclass) holds Settings
                            bootstrap() → ApplicationContext
```

---

## `src/rag_ingestion/domain/`

The innermost layer. No imports from Application, Infrastructure, or Interface.

```
domain/
├── __init__.py           # Re-exports ALL entities + protocols (see below)
│
├── entities/
│   ├── __init__.py       # Re-exports: Document, ParsedDocument,
│   │                       NormalizedDocument, Chunk, ChunkMetadata,
│   │                       EmbeddedChunk
│   ├── document.py       # Document        — raw source file (path, ext, hash, bytes)
│   │                       ParsedDocument  — Docling output (text, title, num_pages)
│   │                       NormalizedDocument — cleaned text + sections tuple
│   └── chunk.py          # ChunkMetadata   — provenance (hash, path, idx, heading, page,
│                           │                              char_start, char_end)
│                           Chunk           — chunk_id, text, metadata
│                           EmbeddedChunk   — chunk + embedding: tuple[float,...]
│
└── protocols/
    ├── __init__.py       # Re-exports: Loader, Parser, Normalizer,
    │                       Chunker, MetadataEnricher, Embedder, VectorStore
    ├── loader.py         # Loader(Protocol) — load(source: str) → list[Document]
    ├── parser.py         # Parser(Protocol) — parse(document: Document) → ParsedDocument
    ├── normalizer.py     # Normalizer(Protocol) — normalize(document: ParsedDocument) → NormalizedDocument
    ├── chunker.py        # Chunker(Protocol) — chunk(document: NormalizedDocument) → list[Chunk]
    ├── metadata_enricher.py # MetadataEnricher(Protocol) — enrich(chunks: list[Chunk]) → list[Chunk]
    ├── embedder.py       # Embedder(Protocol) — embed(chunks: list[Chunk]) → list[EmbeddedChunk]
    └── vector_store.py   # VectorStore(Protocol) — upsert(list[EmbeddedChunk]) → None
                            #                        delete_by_document(hash) → int
                            #                        count() → int
                            # All protocols are @runtime_checkable
```

---

## `src/rag_ingestion/application/`

Coordinates use cases. Depends on Domain only.

```
application/
├── __init__.py           # Layer docstring; __all__ = []
├── services/
│   └── __init__.py       # Stub — planned: IngestionService, MetadataService,
│                           DocumentHashService, ChunkHashService
└── orchestrators/
    └── __init__.py       # Stub — planned: IngestionOrchestrator
                            (Load → Parse → Normalize → Chunk → Embed → Store)
```

> **Status:** No implementations yet. `__init__.py` stubs only.

---

## `src/rag_ingestion/infrastructure/`

Concrete adapters. Depends on Domain + Application.


```
infrastructure/
├── __init__.py             # Layer docstring; __all__ = []
├── loaders/
│   ├── __init__.py         # Exports FileSystemLoader
│   └── filesystem.py       # FileSystemLoader — validates ext, reads bytes,
│                           #   computes hash, produces Document; no parsing
├── parsers/
│   ├── __init__.py         # Exports DoclingParser
│   └── docling_parser.py   # DoclingParser — wraps DocumentConverter,
│                           #   implements Parser protocol, surfaces failures
│                           #   as ValueError / RuntimeError with source path
├── normalizers/
│   ├── __init__.py         # Exports DefaultDocumentNormalizer
│   └── default.py          # DefaultDocumentNormalizer — normalizes whitespace,
│                           #   line endings, encoding, and extracts sections
├── metadata_enrichers/
│   ├── __init__.py         # Exports DefaultMetadataEnricher
│   └── default.py          # DefaultMetadataEnricher — adds deterministic
│                           #   document/chunk hashes, timestamps, titles,
│                           #   source file names, and section hierarchy
├── chunkers/
│   ├── __init__.py         # Exports RecursiveChunker
│   └── recursive.py        # RecursiveChunker — chunker respecting CHUNK_SIZE
│                           #   and CHUNK_OVERLAP, preserving sections
├── embedders/
│   └── __init__.py         # Stub — planned: SentenceTransformerEmbedder
│                           (loads model, batches, optionally normalizes vectors)
└── storage/
    └── __init__.py         # Stub — planned: ChromaVectorStore
                            (persistent ChromaDB, upsert/delete/count)
```

---

## `src/rag_ingestion/interface/`

Entry points. Depends on Application + Domain via composition root.

```
interface/
├── __init__.py           # Layer docstring; __all__ = []
└── cli/
    ├── __init__.py       # CLI package docstring; planned commands: ingest, validate
    └── main.py           # build_parser() → ArgumentParser
                            main() → int   (entry point: `rag-ingestion` script)
                            --version, --config flags implemented
                            --config prints Settings JSON to stdout
```

---

## `tests/`

```
tests/
├── unit/
│   ├── config.py              # 6 tests — Settings defaults, extension normalization,
│   │                            chunk config, collection name, persist dir, hash algo
│   ├── test_bootstrap.py      # 3 tests — bootstrap() returns ApplicationContext,
│   │                            uses cached settings, correct defaults
│   ├── test_logging.py        # 4 tests — logger creation, bound context,
│   │                            json format, console format
│   ├── domain/
│   │   ├── __init__.py        # Package marker
│   │   ├── test_entities.py   # 20 tests — Document, ParsedDocument,
│   │   │                        NormalizedDocument, ChunkMetadata, Chunk,
│   │   │                        EmbeddedChunk: fields, frozen, equality
│   │   └── test_protocols.py  # 20 tests — isinstance checks (runtime_checkable),
│   │                            negative conformance, behavioural contracts for all
│   │                            7 protocols using in-module stubs
│   └── infrastructure/
│       ├── __init__.py        # Package marker
│       ├── loaders/
│       │   ├── __init__.py    # Package marker
│       │   └── test_filesystem_loader.py
│       │                      # 14 tests — protocol conformance, init config,
│       │                      #   validation errors, happy-path loads, filtering,
│       │                      #   hash correctness, case-insensitive ext matching,
│       │                      #   empty files, frozen Document assertion
│       └── parsers/
│           ├── __init__.py    # Package marker
│           └── test_docling_parser.py
│                              # 26 tests — protocol conformance, init config,
│                              #   happy-path conversion, all 7 supported extensions,
│                              #   metadata (title, num_pages) preservation,
│                              #   title fallback, zero-page unpaginated formats,
│                              #   RuntimeError on converter failure (+ chaining),
│                              #   empty-text fallback, ValueError for unsupported
│                              #   extensions, converter not called on rejection
│       ├── chunkers/
│       │   ├── __init__.py    # Package marker
│       │   └── test_recursive_chunker.py
│       │                      # 7 tests — protocol conformance, configurable chunk size,
│       │                      #   overlap, section preservation, deterministic IDs,
│       │                      #   empty documents, recursive split
│       ├── metadata_enrichers/
│       │   └── test_default.py
│       │                      # 7 tests — protocol conformance, deterministic hashes,
│       │                      #   metadata preservation, ordering, timestamp fallback
│       └── normalizers/
│           ├── __init__.py    # Package marker
│           └── test_default.py
│                              # 8 tests — protocol conformance, whitespace,
│                              #   unicode (NFC), repeated blank lines,
│                              #   section preservation, empty document handling,
│                              #   deterministic output, line endings
└── integration/               # Empty — integration tests added only when requested
```

---

## Configuration reference (`config.py` → `.env`)

| Env var                   | Default                              | Type          |
|---------------------------|--------------------------------------|---------------|
| `APP_NAME`                | `rag-ingestion`                      | str           |
| `APP_ENV`                 | `development`                        | Literal       |
| `APP_DEBUG`               | `false`                              | bool          |
| `LOG_LEVEL`               | `INFO`                               | Literal       |
| `LOG_FORMAT`              | `json`                               | Literal       |
| `SUPPORTED_EXTENSIONS`    | `.pdf,.docx,.txt,.md,.markdown,.html,.htm` | str    |
| `CHUNK_SIZE`              | `512`                                | int (≥64)     |
| `CHUNK_OVERLAP`           | `64`                                 | int (≥0)      |
| `CHUNKING_STRATEGY`       | `recursive`                          | Literal       |
| `EMBEDDING_MODEL`         | `BAAI/bge-small-en-v1.5`            | str           |
| `EMBEDDING_BATCH_SIZE`    | `32`                                 | int (≥1)      |
| `EMBEDDING_DEVICE`        | `cpu`                                | str           |
| `EMBEDDING_NORMALIZE`     | `true`                               | bool          |
| `CHROMA_PERSIST_DIRECTORY`| `./chroma`                           | Path          |
| `CHROMA_COLLECTION`       | `documents`                          | str           |
| `HASH_ALGORITHM`          | `sha256`                             | Literal       |
| `DOCLING_ENABLE_OCR`      | `false`                              | bool          |
| `MAX_WORKERS`             | `4`                                  | int (≥1)      |
| `DEFAULT_INPUT_DIRECTORY` | `./documents`                        | Path          |

---



## What does NOT exist yet (next commits)

| Component                      | Location                                         |
|--------------------------------|--------------------------------------------------|
| `DoclingParser`                | ✅ implemented                                   |
| `DefaultDocumentNormalizer`    | ✅ implemented                                   |
| `RecursiveChunker`             | ✅ implemented                                   |
| `SentenceTransformerEmbedder`  | `infrastructure/embedders/sentence_transformer.py` |
| `ChromaVectorStore`            | `infrastructure/storage/chroma.py`               |
| `IngestionOrchestrator`        | `application/orchestrators/ingestion.py`         |
| `ingest` CLI command           | `interface/cli/main.py`                          |
