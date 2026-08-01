You are implementing the next Git commit of this repository.

The repository already exists locally.(on current working directory)

DO NOT regenerate existing files.

DO NOT restate the repository tree.

DO NOT explain the architecture unless explicitly requested.

Read REPO_TREE.md first. Skip directory traversal unless investigating a specific bug, unsure about the implementation of any feature or the current implementation of the feature .

Only output:

1. Commit title
2. Files modified (relative paths)
3. Complete contents of NEW files
4. Unified diff (or exact replacement blocks) for EXISTING files
5. New tests
6. Run commands
7. Short commit summary

Assume all previous commits exist exactly as committed.

--------------------------------------------------
PROJECT RULES
--------------------------------------------------



Dependency manager:
- uv
beforestarting to implement go through the project.toml file for the current project dependancies
Architecture:

Domain
↓
Application
↓
Infrastructure
↓
Interface

Dependencies always point inward.

Use constructor injection.

Single composition root.

No DI frameworks.

--------------------------------------------------
LOCKED STACK
--------------------------------------------------

Parsing:
- Docling

Embeddings:
- Sentence Transformers
- configurable model
- default:
  BAAI/bge-small-en-v1.5

Vector DB:
- ChromaDB

Validation:
- Pydantic v2

Configuration:
- pydantic-settings

Logging:
- structlog

Testing:
- pytest
- pytest-mock

--------------------------------------------------
QUALITY RULES
--------------------------------------------------

Never generate:

TODO
pass
...
placeholder implementations

Everything must compile.

No broken imports.

No references to future commits.

No dead code.

No circular imports.

--------------------------------------------------
TESTING
--------------------------------------------------

Every commit includes tests.

Unit tests:

- isolated
- deterministic
- no filesystem
- no network
- use mocks

Integration tests only when requested.

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

## Commit <N>: <title>

### Files Modified

- ...

### New Files

<complete contents>

### Modified Files

<unified diffs only>

### Tests

<new test files or diffs>

### Run
( dont do this step i will do it manually just tell whenever you reach here)
uv sync
uv run pytest

### Summary

<3 bullets maximum>

Stop after completing ONE commit.

Never begin the next commit automatically.

