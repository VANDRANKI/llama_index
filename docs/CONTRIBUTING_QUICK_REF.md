# LlamaIndex Contributor Quick Reference

## Setup

```bash
git clone https://github.com/run-llama/llama_index.git
cd llama_index

# Install core for development
pip install -e "llama-index-core[dev]"
```

## Running tests

```bash
# Unit tests (no network calls)
pytest llama-index-core/tests/unit -v

# Integration tests (requires API keys)
OPENAI_API_KEY=... pytest llama-index-core/tests/integration -v
```

## Adding a new integration

1. Create `llama-index-integrations/<category>/<package-name>/`.
2. Follow the template in `llama-index-integrations/INTEGRATION_TEMPLATE/`.
3. Add tests under `<package>/tests/`.
4. Register the integration in the main `llama-index` package.

## Debugging retrieval

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for LlamaIndex-specific logs only:
logging.getLogger("llama_index").setLevel(logging.DEBUG)
```

## Common patterns

```python
# Always use Settings instead of ServiceContext (deprecated)
from llama_index.core import Settings
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
```
