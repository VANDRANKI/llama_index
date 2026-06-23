# LlamaIndex Developer Workflow

This guide explains how to set up and work within the LlamaIndex monorepo.

## Repository Layout

```
llama_index/
├── llama-index-core/          # Core abstractions and interfaces
├── llama-index-integrations/   # Third-party integrations
├── llama-index-utils/          # Utility libraries
├── llama-index-instrumentation/ # Observability primitives
└── llama-dev/                  # Development tooling
```

## Setup

```bash
# Install llama-dev tooling
pip install llama-dev

# Install a package in editable mode
pip install -e llama-index-core/

# Install with all optional extras
pip install -e llama-index-core/[dev]
```

## Running Tests

```bash
# From root
make test

# For a specific package
cd llama-index-core
pytest tests/ -v

# With coverage
pytest tests/ --cov=llama_index --cov-report=html
```

## Code Standards

### Type Hints

All public APIs must have full type annotations:

```python
from typing import Any, Optional, Sequence
from llama_index.core.schema import Document, NodeWithScore

def retrieve_nodes(
    query: str,
    index: Any,
    *,
    top_k: int = 5,
    filters: Optional[dict[str, Any]] = None,
) -> list[NodeWithScore]:
    """Retrieve the top-k most relevant nodes for a query.

    Args:
        query: The search query string.
        index: A LlamaIndex index instance.
        top_k: Number of results to return.
        filters: Optional metadata filters to apply.

    Returns:
        List of scored document nodes, sorted by relevance.
    """
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)
```

## Adding a New Integration

1. Create a directory under `llama-index-integrations/<category>/<name>/`
2. Follow the existing structure (see any existing integration for reference)
3. Implement the required interface methods with full type hints
4. Write tests covering the happy path and key error conditions
5. Run `make format && make lint` before submitting

## Pre-commit Checks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
