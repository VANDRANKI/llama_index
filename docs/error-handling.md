# Error Handling in LlamaIndex

## Using LlamaIndexOperationError

Wrap your custom index operations with the typed error class to provide
diagnostic context:

```python
from llama_index.core.utils.error_utils import LlamaIndexOperationError

try:
    index.from_documents(docs)
except Exception as exc:
    raise LlamaIndexOperationError(
        "build index from documents",
        cause=exc,
        detail=f"{len(docs)} documents attempted",
    ) from exc
```

## Validating required inputs

```python
from llama_index.core.utils.error_utils import require_non_empty

def query(query_text: str) -> str:
    require_non_empty(query_text, "query_text", "execute query")
    return engine.query(query_text)
```

## Wrapping embedding calls

```python
from llama_index.core.utils.error_utils import wrap_embedding_error

@wrap_embedding_error
def get_embedding(text: str) -> list[float]:
    return embed_model.get_text_embedding(text)
```

If the embedding provider raises any exception, it is automatically
wrapped in `LlamaIndexOperationError("generate embeddings")`.
