# Contributing Quickstart

A fast-path guide for new contributors. For full details see `CONTRIBUTING.md`.

## Environment Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/run-llama/llama_index
cd llama_index
pip install uv

# Install the core package in editable mode
cd llama-index-core
uv pip install -e ".[dev]"
```

## Running Tests

```bash
# Unit tests (no API keys needed)
pytest tests/unit/ -v

# Integration tests for a specific integration
cd llama-index-integrations/llms/llama-index-llms-openai
pytest tests/ -v
```

## Package Structure

```
llama_index/
├── llama-index-core/          # Core framework
├── llama-index-integrations/  # Provider integrations
│   ├── llms/                  # Language model integrations
│   ├── embeddings/            # Embedding model integrations
│   ├── vector_stores/         # Vector database integrations
│   └── readers/               # Document loaders
├── llama-index-packs/         # Agent packs
└── docs/                      # Documentation
```

## Adding a New Integration

1. Create the package directory:
   ```bash
   mkdir -p llama-index-integrations/llms/llama-index-llms-<name>/
   ```
2. Add `pyproject.toml`, `README.md`, and the integration class.
3. Inherit from the appropriate base class in `llama-index-core`
   (e.g., `BaseLLM`, `BaseEmbedding`, `VectorStore`).
4. Implement all abstract methods including async variants where required.
5. Add tests under `tests/` in the integration package.

## Code Style

```bash
# Format with ruff
ruff format .

# Lint
ruff check .

# Type check
mypy llama_index/
```

- Use Google-style docstrings.
- All public methods need type annotations.
- Provide both sync and async variants for IO-bound operations.

## PR Guidelines

- One feature or fix per PR.
- Tests must pass locally before opening a PR.
- Add an entry to `CHANGELOG.md` for user-visible changes.
- Tag the PR with the relevant integration label (e.g., `llm`, `vector-store`).
