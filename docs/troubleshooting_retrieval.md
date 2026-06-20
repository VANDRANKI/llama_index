# Troubleshooting Retrieval in LlamaIndex

A reference for common retrieval and indexing issues.

---

## Document ingestion issues

### Documents are ingested but retrieval returns no results

**Symptoms:** `index.as_retriever().retrieve(query)` returns an empty list
even though documents were added.

**Common causes:**

1. **Embedding model mismatch** — documents were embedded with model A but
   queries use model B. Embeddings must come from the same model.

   ```python
   from llama_index.embeddings.openai import OpenAIEmbedding

   embed_model = OpenAIEmbedding(model="text-embedding-3-small")
   # Use the same embed_model for both ingestion and querying.
   ```

2. **Empty node content** — if the `TextNode.text` field is empty after
   parsing, the vector store may skip it or return all-zero embeddings.

   ```python
   for node in nodes:
       assert node.get_content(), f"Node {node.node_id} has empty content!"
   ```

3. **Similarity threshold too high** — the default `similarity_top_k` may
   return results that fall below the score threshold of your vector store.

   ```python
   retriever = index.as_retriever(similarity_top_k=10)
   nodes = retriever.retrieve("my query")
   for n in nodes:
       print(n.score, n.get_content()[:80])
   ```

---

### `ValueError: Nodes list is empty` during index construction

The document parser returned zero nodes. Check:

- The file path is correct and the file is not empty.
- The `SimpleDirectoryReader` is pointed at the right directory.
- The file extension is supported (add a custom reader for unsupported types).

---

## Query engine issues

### Responses are irrelevant or hallucinated

1. **Increase `similarity_top_k`** to provide more context chunks.
2. **Enable `response_mode="compact"`** to consolidate chunks before synthesis.
3. **Inspect retrieved nodes** to confirm relevant documents are being fetched:

   ```python
   from llama_index.core import get_response_synthesizer
   from llama_index.core.query_engine import RetrieverQueryEngine

   retriever = index.as_retriever(similarity_top_k=5)
   nodes = retriever.retrieve(query_str)
   print(f"Retrieved {len(nodes)} nodes")
   for n in nodes:
       print(f"Score: {n.score:.3f} | {n.get_content()[:100]}")
   ```

---

## Vector store issues

### `AzureAISearchVectorStore`: falsy metadata raises `AttributeError`

Metadata fields that evaluate to falsy (e.g. `0`, `False`, `""`) were
incorrectly filtered out in earlier versions. Upgrade to the patched release
or pass `filter_falsy_metadata=False` when constructing the store.

---

## Performance

### Ingestion is slow for large document sets

Use the `IngestionPipeline` with `num_workers > 1` to parallelise embedding:

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

pipeline = IngestionPipeline(
    transformations=[SentenceSplitter(), embed_model],
    vector_store=vector_store,
    num_workers=4,
)
nodes = pipeline.run(documents=documents, show_progress=True)
```
