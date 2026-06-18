# LlamaIndex Development Guide

## Repository Structure

```
llama-index-core/          # Core abstractions (most contributors work here)
llama-index-integrations/  # Provider-specific integrations
llama-index-utils/         # Shared utilities
llama-index-instrumentation/ # Observability
llama-dev/                 # Development tooling
```

## Setup

```bash
git clone https://github.com/VANDRANKI/llama_index.git
cd llama_index

# Install core in editable mode
cd llama-index-core
pip install -e ".[dev]"
cd ..
```

For an integration package:

```bash
cd llama-index-integrations/llms/llama-index-llms-openai
pip install -e ".[dev]"
```

## Running Tests

```bash
# From inside the package directory:
pytest tests/ -v

# Integration tests require API keys:
export OPENAI_API_KEY=sk-...
pytest tests/llms/ -v -m integration
```

## Code Quality

```bash
# From any package directory:
make format  # ruff format + isort
make lint    # ruff check + mypy
```

## Writing a New Integration

1. Copy the template from `llama-dev/`
2. Implement the required protocol methods with full type hints
3. Add a `README.md` to the package root
4. Add unit tests using `unittest.mock` for any API calls
5. Add an integration test tagged with `@pytest.mark.integration`

## Type Hints

All public methods require complete type annotations. Use `Optional[X]`
(not `X | None`) for Python 3.9 compatibility:

```python
from typing import Any, Optional, Sequence

class BaseLLM:
    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        ...
```

## Docstrings

Use Google-style. Do not include types in the docstring body —
they are already in the signature.

## Commit Convention

```
feat(core): add async streaming to BaseLLM
fix(openai): handle token count in streaming mode
docs(integrations): add usage example for Cohere reranker
```
