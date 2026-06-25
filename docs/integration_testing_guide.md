# Integration Testing Guide

This guide explains how to write integration tests for LlamaIndex components.

## Test Organization

```
tests/
├── unit/          # No network calls, fast, always run in CI
└── integration/   # Real API calls, require environment variables
    ├── conftest.py  # Shared fixtures
    └── test_<component>.py
```

Keep unit and integration tests strictly separated. Unit tests must never
make network calls.

## Writing Integration Tests

```python
# tests/integration/test_vector_store_retrieval.py
import os

import pytest
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding


@pytest.fixture(scope="module")
def openai_embed_model():
    """Reuse the embedding model across tests in this module."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAIEmbedding(model="text-embedding-3-small")


@pytest.fixture(scope="module")
def sample_index(openai_embed_model, tmp_path_factory):
    """Build a real index once and share it across tests."""
    data_dir = tmp_path_factory.mktemp("data")
    (data_dir / "sample.txt").write_text(
        "LlamaIndex is a data framework for LLM applications."
    )
    documents = SimpleDirectoryReader(str(data_dir)).load_data()
    return VectorStoreIndex.from_documents(
        documents, embed_model=openai_embed_model
    )


def test_basic_retrieval(sample_index):
    """Verify that retrieval returns relevant nodes."""
    retriever = sample_index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve("What is LlamaIndex?")

    assert len(nodes) >= 1
    assert "LlamaIndex" in nodes[0].text


def test_query_engine_response(sample_index):
    """Verify that a query engine produces a non-empty response."""
    engine = sample_index.as_query_engine()
    response = engine.query("What does LlamaIndex do?")

    assert response.response
    assert len(response.response) > 10
```

## Running Integration Tests

```bash
# Run only integration tests
pytest tests/integration/ -v

# Run with a specific marker
pytest -m integration -v

# Skip slow tests
pytest tests/integration/ -v -m "not slow"
```

## Fixture Scopes for Expensive Resources

| Scope | Recreated | Use For |
|---|---|---|
| `function` | Every test | Cheap resources, isolated tests |
| `module` | Once per file | LLM clients, embedding models |
| `session` | Once per run | Persistent stores, loaded indexes |

## Handling Missing Credentials

Always skip gracefully rather than fail hard:

```python
@pytest.fixture
def anthropic_llm():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    from llama_index.llms.anthropic import Anthropic
    return Anthropic(model="claude-sonnet-4-6")
```

This ensures CI does not fail when optional provider credentials are absent.
