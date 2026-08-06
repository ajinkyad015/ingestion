# Repository Tree — `rag-ingestion`

<!-- markdownlint-disable -->

> **Purpose of this file:** Give any agent an instant, token-efficient snapshot of the
> current repo state. Read this file first; skip full directory traversal.
>
> **Maintainance rule:** Update this file in every commit that adds, removes, or
> meaningfully changes a file.

---

## Last updated

**Commit 10 (inferred) — "Add integration tests"**
Local time: 2026-08-06 · Python ≥ 3.12.13 · uv · hatchling

> Synced directly from a repository snapshot (`ingestion-main__2_.zip`); no `.git`
> history was available in the snapshot, so the commit number/title above is inferred
> from the diff against the previous tree (Commit 8) rather than read from git log.
> Correct it if your local history disagrees.

---

## Stack (locked)

| Concern        | Library                          |
|----------------|-----------------------------------|
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

- **Full pipeline is wired end-to-end**: `Loader → Parser → Normalizer → Chunker →
  MetadataEnricher → Embedder → VectorStore`, coordinated by `IngestionOrchestrator`.
- Implemented dependency chain: `Settings` → `FileSystemLoader` → `Document`
  → `DoclingParser` → `ParsedDocument` → `DefaultDocumentNormalizer`
  → `NormalizedDocument` → `RecursiveChunker` → `Chunk` → `DefaultMetadataEnricher`
  → `SentenceTransformerEmbedder` → `EmbeddedChunk` → `ChromaVectorStore`;
  `IngestionOrchestrator` composes all seven stages behind `ingest(source) -> IngestionResult`.
- `ChromaVectorStore` (`infrastructure/storage/chroma.py`) now implements the
  `VectorStore` protocol: lazy `chromadb.PersistentClient`, `get_or_create_collection`
  (cosine space), `upsert`/`delete_by_document`/`count`.
- `IngestionOrchestrator` (`application/orchestrators/ingestion.py`) is implemented:
  per-document try/except isolates stage failures into `IngestionFailure` records
  (`KeyboardInterrupt`/`SystemExit` still propagate immediately, as do Loader-level
  failures), structured logging (`structlog`) fires at every stage transition, and the
  run returns an immutable `IngestionResult` (counters + failures + elapsed time).
- `bootstrap()` now composes and returns the **fully wired** `ApplicationContext`
  (`settings`, `loader`, `parser`, `normalizer`, `chunker`, `metadata_enricher`,
  `embedder`, `vector_store`, `orchestrator`) — previously it stopped at the embedder.
- `interface/cli/main.py` now implements the `ingest` subcommand end-to-end: resolves
  the target directory (positional arg or `DEFAULT_INPUT_DIRECTORY`), calls
  `context.orchestrator.ingest(target)`, prints a run summary (documents/chunks/
  embeddings/vectors/failures/elapsed), and lists any failed documents with stage + error.
- `application/services/` remains **stub-only** — `IngestionService`, `MetadataService`,
  `DocumentHashService`, `ChunkHashService` are still just planned names in the module
  docstring; their responsibilities currently live inline in the orchestrator and the
  metadata enricher.
- Test suite now covers all seven infrastructure adapters, both entity/protocol modules,
  `bootstrap`, `logging`, and the orchestrator (178 test functions across 13 files — see
  the `tests/` section below for the full breakdown and a discovery caveat).
- `tests/integration/` **does exist** in this snapshot; added `test_pipeline.py` with end-to-end integration tests using real implementations.
- For new features, keep the dependency order in mind: Domain stays dependency-free,
  Infrastructure plugs into Domain protocols, Application orchestrates, and Interface
  only composes through `bootstrap.py`.

---

## Root

```
d:\ingestion\
├── .env.example          # All env-var keys with documented defaults
├── .gitignore
├── pyproject.toml        # Build config, deps, pytest/ruff/mypy settings
├── uv.lock               # Locked dependency graph (181 packages)
├── README.md             # Project overview
├── REPO_TREE.md          # ← THIS FILE
└── docs/
    ├── INSTRUCTIONS.md    # Agent operating instructions for this repo: commit/output
    │                        format, architecture + testing rules, locked stack.
    │                        Not application code; governs how commits are produced.
    └── REPO_TREE.md       # Mirror of repo tree
```

> **Sync note:** the previous tree snapshot (Commit 8) also listed `.python-version`
> and `README.pdf` at repo root. Neither file is present in the archive this sync was
> generated from, so they've been removed above. `INSTRUCTIONS.md` *is* present in this
> snapshot and wasn't previously listed — added above. If `.python-version` /
> `README.pdf` still exist in your working tree, this is a snapshot gap, not evidence
> they were deleted — re-add them on your next sync if so.

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
                            ApplicationContext(frozen dataclass) holds: settings,
                            loader, parser, normalizer, chunker, metadata_enricher,
                            embedder, vector_store, orchestrator
                            bootstrap() → ApplicationContext (instantiates every
                            adapter plus IngestionOrchestrator, in that order)
```

---

## `src/rag_ingestion/domain/`

The innermost layer. No imports from Application, Infrastructure, or Interface.
**Unchanged since Commit 8.**

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
│                           DocumentHashService, ChunkHashService (still not implemented)
└── orchestrators/
    ├── __init__.py       # Re-exports IngestionFailure, IngestionOrchestrator, IngestionResult
    └── ingestion.py       # IngestionFailure (frozen)   — document_path, stage, error
                            # IngestionResult  (frozen)   — documents_processed,
                            #   chunks_created, embeddings_created, vectors_written,
                            #   failures: tuple[IngestionFailure, ...], elapsed_time_seconds
                            # IngestionOrchestrator(loader, parser, normalizer, chunker,
                            #   metadata_enricher, embedder, vector_store) — constructor-
                            #   injected, depends only on Domain protocols.
                            #   .ingest(source) → IngestionResult:
                            #     Loader.load(source) — pipeline failure, re-raises
                            #     per document: Parser → Normalizer → Chunker → (skip
                            #       enrich/embed/store + still count as processed if
                            #       zero chunks) → MetadataEnricher → Embedder (1 call/
                            #       doc) → VectorStore.upsert (1 call/doc)
                            #     per-document exceptions caught → IngestionFailure,
                            #       processing continues with the next document;
                            #       KeyboardInterrupt/SystemExit always re-raise
                            #     structlog events at every stage transition
                            #       (pipeline_started, document_loaded, document_parsed,
                            #       document_normalized, document_chunked,
                            #       metadata_enriched, embeddings_created,
                            #       vectors_written, document_failed, pipeline_completed)
```

> **Status:** `IngestionOrchestrator` is fully implemented and tested.
> `application/services/` remains stub-only (`__init__.py` docstring only).

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
│   ├── __init__.py         # Exports SentenceTransformerEmbedder
│   └── sentence_transformer.py # SentenceTransformerEmbedder — lazy model load,
│                           #   batched encode, optional normalization,
│                           #   runtime error wrapping
└── storage/
    ├── __init__.py         # Exports ChromaVectorStore
    └── chroma.py           # ChromaVectorStore — lazy chromadb.PersistentClient +
                            #   get_or_create_collection (hnsw:space=cosine);
                            #   upsert(ids/documents/embeddings/metadatas) — raises
                            #   ValueError on empty input; delete_by_document(hash)
                            #   (collection.get(where=...) → delete by ids, returns
                            #   count); count() → collection.count()
```

> **Status:** All seven infrastructure adapters (`loaders`, `parsers`, `normalizers`,
> `metadata_enrichers`, `chunkers`, `embedders`, `storage`) are implemented. `storage/`
> was the last remaining stub as of Commit 8 and is now `ChromaVectorStore`.

---

## `src/rag_ingestion/interface/`

Entry points. Depends on Application + Domain via composition root.

```
interface/
├── __init__.py           # Layer docstring; __all__ = []
└── cli/
    ├── __init__.py       # CLI package docstring; planned commands: ingest, validate, version
    └── main.py           # build_parser() → ArgumentParser
                            #   --version, --config, subcommand `ingest [source]`
                            # main() → int   (entry point: `rag-ingestion` script)
                            #   --config: prints Settings JSON (context.settings
                            #     .model_dump_json) to stdout
                            #   ingest: resolves target = args.source or
                            #     settings.default_input_directory; calls
                            #     context.orchestrator.ingest(target); prints a run
                            #     summary (documents processed / chunks created /
                            #     embeddings created / vectors written / failures /
                            #     elapsed seconds) and, if any failures occurred,
                            #     lists each as "- {path} [{stage}] {error}"
                            #   no subcommand: prints --help and returns 0
```

> **Status:** `ingest` is implemented and wired to `bootstrap()`/`IngestionOrchestrator`.
> `validate` remains only a planned command name in the package docstring.

---

## `tests/`

```
tests/
├── unit/
│   ├── config.py              # 6 tests — Settings defaults, extension normalization,
│   │                            chunk config, collection name, persist dir, hash algo
│   │                            ⚠ filename does not match pytest's default
│   │                            `test_*.py` / `*_test.py` glob, so these 6 tests are
│   │                            NOT collected by a plain `uv run pytest` — pre-existing
│   │                            gap, unchanged since Commit 8, flagged here for visibility
│   ├── test_bootstrap.py      # 3 tests — bootstrap() returns ApplicationContext,
│   │                            uses cached settings, correct defaults
│   ├── test_logging.py        # 4 tests — logger creation, bound context,
│   │                            json format, console format
│   │
│   ├── application/                       # no __init__.py
│   │   └── orchestrators/                 # no __init__.py
│   │       └── test_ingestion.py
│   │                          # 40 tests — IngestionFailure/IngestionResult frozen +
│   │                          #   tuple invariants (TestResultModels); happy-path
│   │                          #   counters for single/multiple documents, empty
│   │                          #   loader, and zero-chunk documents
│   │                          #   (TestIngestionOrchestratorHappyPath); stage-call
│   │                          #   ordering load→parse→normalize→chunk→enrich→embed→
│   │                          #   store, loader receives the source, enricher output
│   │                          #   (not raw chunks) reaches the embedder
│   │                          #   (TestPipelineOrdering); per-document failures at
│   │                          #   every stage recorded + pipeline continues to the
│   │                          #   next document (TestDocumentFailures); loader
│   │                          #   (pipeline-level) errors and KeyboardInterrupt/
│   │                          #   SystemExit propagate immediately
│   │                          #   (TestPipelineFailures); structlog event coverage
│   │                          #   for every stage + failure path (TestLogging);
│   │                          #   embedder/vector_store call-count invariants on
│   │                          #   zero-chunk and mixed success/failure runs
│   │                          #   (TestCallCountInvariants)
│   │
│   ├── domain/
│   │   ├── __init__.py        # Package marker
│   │   ├── test_entities.py   # 22 tests — Document, ParsedDocument,
│   │   │                        NormalizedDocument, ChunkMetadata, Chunk,
│   │   │                        EmbeddedChunk: fields, frozen, equality
│   │   └── test_protocols.py  # 24 tests — isinstance checks (runtime_checkable),
│   │                            negative conformance, behavioural contracts for all
│   │                            7 protocols using in-module stubs
│   │
│   └── infrastructure/
│       ├── __init__.py        # Package marker
│       ├── chunkers/
│       │   ├── __init__.py    # Package marker
│       │   └── test_recursive_chunker.py
│       │                      # 7 tests — protocol conformance, empty document,
│       │                      #   document smaller than chunk size, section-hierarchy
│       │                      #   preservation + deterministic chunk IDs, recursive
│       │                      #   split of an oversized section, deterministic chunk
│       │                      #   boundaries, configurable overlap
│       ├── embedders/                     # no __init__.py
│       │   └── test_sentence_transformer.py
│       │                      # 11 tests — protocol conformance, lazy model init,
│       │                      #   settings wiring, batching + order preservation,
│       │                      #   normalization flag (parametrized true/false),
│       │                      #   metadata preserved unchanged, model cached per
│       │                      #   instance, empty-input fast path (no model load),
│       │                      #   load-failure and inference-failure wrapping
│       ├── loaders/
│       │   ├── __init__.py    # Package marker
│       │   └── test_filesystem_loader.py
│       │                      # 18 tests — protocol conformance, init config,
│       │                      #   validation errors, happy-path loads, filtering,
│       │                      #   hash correctness, case-insensitive ext matching,
│       │                      #   empty files, frozen Document assertion
│       ├── metadata_enrichers/            # no __init__.py
│       │   └── test_default_mdata.py
│       │                      # 7 tests — protocol conformance, order preserved /
│       │                      #   input not mutated, metadata attached + existing
│       │                      #   values preserved, deterministic hashes for
│       │                      #   identical input, chunk-text change isolates
│       │                      #   chunk_hash only, doc change alters document_hash,
│       │                      #   timestamps loaded from filesystem when missing
│       ├── normalizers/                   # no __init__.py
│       │   └── test_default.py
│       │                      # 8 tests — protocol conformance, whitespace,
│       │                      #   unicode (NFC), repeated blank lines,
│       │                      #   section preservation, empty document handling,
│       │                      #   deterministic output, line endings
│       ├── parsers/
│       │   ├── __init__.py    # Package marker
│       │   └── test_docling_parser.py
│       │                      # 23 tests — protocol conformance, init config,
│       │                      #   happy-path conversion, all 7 supported extensions,
│       │                      #   metadata (title, num_pages) preservation,
│       │                      #   title fallback, zero-page unpaginated formats,
│       │                      #   RuntimeError on converter failure (+ chaining),
│       │                      #   empty-text fallback, ValueError for unsupported
│       │                      #   extensions, converter not called on rejection
│       └── storage/                       # no __init__.py
│           └── test_chroma_vector_store.py
│                              # 5 tests — protocol conformance, upsert requires
│                              #   ≥1 chunk (ValueError), upsert persists ids/
│                              #   documents/embeddings/metadata onto a fake
│                              #   collection, delete_by_document returns the
│                              #   deleted count, count() returns collection count
│
└── integration/
    └── test_pipeline.py       # 4 tests — end-to-end pipeline success,
                               #   determinism, failure handling, cli integration
```

> **Test file/dir naming is inconsistent**: some infrastructure subpackages have an
> `__init__.py` package marker (`chunkers/`, `loaders/`, `parsers/`) and some don't
> (`embedders/`, `metadata_enrichers/`, `normalizers/`, `storage/`), and neither
> `tests/unit/application/` nor `tests/unit/application/orchestrators/` has one either.
> This doesn't currently break collection (pytest's default `rootdir`-relative import
> mode tolerates it as long as module basenames stay unique) but is worth normalizing
> in a future cleanup commit.

> **Total: 182 test functions across 14 files** (176 collected by a default
> `uv run pytest` run — the 6 in `tests/unit/config.py` are excluded from discovery;
> see the note on that file above).

---

## Configuration reference (`config.py` → `.env`)

| Env var                   | Default                              | Type          |
|----------------------------|----------------------------------------|-----------------|
| `APP_NAME`                | `rag-ingestion`                      | str           |
| `APP_ENV`                 | `development`                        | Literal       |
| `APP_DEBUG`                | `false`                              | bool          |
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

*(Unchanged since Commit 8 — no config fields were added or modified.)*

---

## What does NOT exist yet (next commits)

| Component                                  | Status / Location                                          |
|---------------------------------------------|--------------------------------------------------------------|
| `DoclingParser`                             | ✅ implemented                                               |
| `DefaultDocumentNormalizer`                 | ✅ implemented                                               |
| `RecursiveChunker`                          | ✅ implemented                                               |
| `SentenceTransformerEmbedder`               | ✅ implemented                                               |
| `ChromaVectorStore`                         | ✅ implemented — `infrastructure/storage/chroma.py`          |
| `IngestionOrchestrator`                     | ✅ implemented — `application/orchestrators/ingestion.py`    |
| `ingest` CLI command                        | ✅ implemented — `interface/cli/main.py`                     |
| `application/services/*`                    | ❌ stub only — `IngestionService`, `MetadataService`, `DocumentHashService`, `ChunkHashService` |
| `validate` CLI command                      | ❌ planned name only, no implementation                      |
| `tests/integration/`                        | ✅ implemented — `test_pipeline.py`                          |
| Additional vector store adapters            | ❌ roadmap only (pgvector, Qdrant, Weaviate, Milvus, Pinecone) — see README.md |
| Additional embedding providers              | ❌ roadmap only — see README.md                               |
| PPTX / XLSX / CSV / email ingestion         | ❌ roadmap only — see README.md                               |
| Hybrid retrieval / reranking                | ❌ roadmap only — see README.md                               |

<!-- markdownlint-enable -->
