# Retrieval Pipeline Patterns

This guide covers common patterns for building retrieval pipelines with
LlamaIndex, from basic vector search to advanced hybrid retrieval and
metadata filtering.

## Basic RAG Pipeline

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine

# Load and index documents
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Build a query engine (wraps retriever + response synthesiser)
query_engine = index.as_query_engine(
    similarity_top_k=5,  # retrieve top-5 most similar chunks
)

response = query_engine.query("What are the key findings?")
print(response)
```

---

## Retriever-Only Mode

Sometimes you want raw retrieved nodes without synthesis — for re-ranking,
logging, or feeding into a custom LLM prompt:

```python
retriever = index.as_retriever(similarity_top_k=10)
nodes = retriever.retrieve("climate change impacts")

for node in nodes:
    print(f"Score: {node.score:.3f} | Source: {node.metadata.get('file_name')}")
    print(node.text[:200])
    print()
```

---

## Metadata Filtering

Narrow the search space at retrieval time using metadata filters:

```python
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="year", value="2024"),
        MetadataFilter(key="category", value="finance"),
    ]
)

retriever = index.as_retriever(
    similarity_top_k=5,
    filters=filters,
)
nodes = retriever.retrieve("quarterly earnings")
```

> Metadata filters are supported by most vector store integrations
> (Chroma, Qdrant, Pinecone, Weaviate). Check your integration's docs
> for supported filter operators.

---

## Hybrid Retrieval (Vector + BM25)

Combine dense vector search with sparse BM25 keyword search for better
recall on exact-match queries:

```python
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore,
    similarity_top_k=5,
)

vector_retriever = index.as_retriever(similarity_top_k=5)

hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=3,  # generate query variants for fusion
    mode="reciprocal_rerank",
)

nodes = hybrid_retriever.retrieve("LLM inference optimisation techniques")
```

---

## Node Post-Processing (Re-ranking)

Apply post-processors after retrieval to improve result quality:

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3,  # keep top-3 after reranking
)

query_engine = RetrieverQueryEngine(
    retriever=index.as_retriever(similarity_top_k=10),
    node_postprocessors=[reranker],
)

response = query_engine.query("What are the limitations of chain-of-thought?")
```

---

## Async Retrieval

```python
import asyncio

async def retrieve_async(query: str) -> list:
    nodes = await retriever.aretrieve(query)
    return nodes

async def query_async(query: str) -> str:
    response = await query_engine.aquery(query)
    return str(response)

# Run both concurrently
results = asyncio.run(
    asyncio.gather(
        query_async("What is RAG?"),
        query_async("Explain fine-tuning vs in-context learning."),
    )
)
```

---

## Choosing `similarity_top_k`

| Use Case | Recommended `similarity_top_k` |
|----------|---------------------------------|
| Precise factual QA | 3–5 |
| Exploratory research | 8–15 |
| Long-form synthesis | 10–20 |
| Reranked pipeline | 10–20 (then re-rank to 3–5) |

Higher values improve recall but increase LLM context window usage and cost.
Use a reranker when you need both high recall and high precision.

---

## Common Pitfalls

- **Chunk size too large**: Chunks above ~1000 tokens dilute relevance scores
  and waste context. Use 512–800 tokens with a 50–100 token overlap.
- **Missing metadata**: Filters fail silently when documents lack the filtered
  key. Always verify metadata during ingestion.
- **Not streaming**: For long synthesis, use `query_engine.query(..., streaming=True)`
  to surface tokens as they are generated.
- **Forgetting `node.score`**: After retrieval, `node.score` is the cosine
  similarity. Logging scores helps debug poor retrieval quality.
