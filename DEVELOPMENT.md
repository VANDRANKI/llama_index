# Development Guide

This guide supplements [`CONTRIBUTING.md`](./CONTRIBUTING.md) with practical, day-to-day development workflows for the LlamaIndex monorepo.

---

## Prerequisites

- Python 3.10 or newer (CI tests against 3.10, 3.11, and 3.12)
- [`uv`](https://docs.astral.sh/uv/) — used for all dependency management
- `git`

---

## Monorepo Structure

```text
llama_index/
├── llama-index-core/                   # Core abstractions, base classes, pipelines
│   └── llama_index/core/
├── llama-index-integrations/           # Provider-specific integrations
│   ├── llms/                           # LLM providers (openai, anthropic, ...)
│   ├── embeddings/                     # Embedding models
│   ├── vector_stores/                  # Vector store adapters
│   ├── readers/                        # Data loaders / readers
│   └── ...
├── llama-index-instrumentation/        # Tracing and observability hooks
├── llama-index-utils/                  # Shared utility helpers
├── llama-index-packs/                  # High-level, opinionated "packs"
├── llama-index-experimental/           # Experimental / unstable features
└── docs/                              # Documentation source
```

Each top-level directory is an **independently versioned Python package** with its own `pyproject.toml`. You should only install and test the package(s) you are modifying.

---

## Per-Package Installation

`uv` handles creating and populating the virtual environment for each package. Run all commands from within the package directory:

```bash
# Example: working on the OpenAI LLM integration
cd llama-index-integrations/llms/llama-index-llms-openai

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

For **core**:

```bash
cd llama-index-core
uv pip install -e ".[dev]"
```

---

## Running Tests

Run tests from inside the target package directory:

```bash
# Run all tests for the current package
uv run -- pytest

# Run a specific test file
uv run -- pytest tests/test_vector_store.py -v

# Filter by test name
uv run -- pytest -k "test_similarity_search" -v

# Skip integration tests that require external API keys
uv run -- pytest -m "not integration"
```

Coverage must be **at least 50%** per package or CI will fail.

---

## Linting and Formatting

All code is checked with `ruff` and `mypy` via pre-commit hooks.

### Install hooks (once, from the repo root)

```bash
uv sync
uv run pre-commit install
```

### Run all checks manually

```bash
# From repo root — checks all staged/changed files
uv run pre-commit run --all-files

# Or target a specific package
uv run make lint
```

### Type-checking compatibility note

The codebase targets **Python 3.10**. Always use `Optional[X]` instead of `X | None` for type annotations — the `|` union syntax requires Python 3.10+ at *runtime* and fails in 3.9 environments still covered by some downstream users.

```python
# Correct — works on Python 3.9+
from typing import Optional
def my_func(x: Optional[str] = None) -> Optional[int]: ...

# Avoid — runtime error on Python <3.10
def my_func(x: str | None = None) -> int | None: ...
```

---

## Finding the Right Package to Modify

See the ["Finding the right package"](./CONTRIBUTING.md#finding-the-right-package) section in `CONTRIBUTING.md` for a decision guide on where changes belong.

Quick reference:

| Change type | Package |
|-------------|--------|
| Base classes, interfaces, pipelines | `llama-index-core` |
| LLM provider support | `llama-index-integrations/llms/llama-index-llms-<provider>` |
| Embedding model support | `llama-index-integrations/embeddings/llama-index-embeddings-<provider>` |
| Vector store adapter | `llama-index-integrations/vector_stores/llama-index-vector-stores-<name>` |
| Data loading / readers | `llama-index-integrations/readers/llama-index-readers-<name>` |
| Tracing / callbacks | `llama-index-instrumentation` |
| Shared utilities | `llama-index-utils` |

---

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(core): add async support to VectorStoreIndex.from_documents
fix(llms-openai): handle rate limit retry with exponential backoff
docs(embeddings): add example for batch embed_documents call
refactor(core): extract _build_index_from_nodes helper
chore: bump ruff to 0.4.0
```

---

## Useful Make Targets

From the repo root:

| Target | Description |
|--------|-------------|
| `make lint` | Run ruff + mypy across all packages |
| `make format` | Apply ruff formatter |
| `make test` | Run tests for the current package |
