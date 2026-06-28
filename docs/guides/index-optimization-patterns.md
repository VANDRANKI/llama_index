# Index Optimization Patterns

This guide covers strategies for improving retrieval speed, accuracy, and
cost-efficiency in LlamaIndex pipelines.

## 1. Choose the right index type

| Index type | Best for | Trade-off |
|---|---|---|
| `VectorStoreIndex` | Semantic similarity search | Requires embedding call per query |
| `SummaryIndex` | Summarisation over entire corpus | Reads all nodes; expensive at scale |
| `KeywordTableIndex` | Exact keyword lookup | No semantic understanding |
| `KnowledgeGraphIndex` | Entity relationship queries | High build cost |

## 2. Chunking strategy

Chunk size directly affects retrieval precision and LLM context usage.

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=512,       # tokens per chunk
    chunk_overlap=50,     # overlap between adjacent chunks
)
nodes = splitter.get_nodes_from_documents(documents)
```

General guidance:
- **Short answers** (FAQ, QA): 256-512 tokens
- **Reasoning over context**: 512-1024 tokens
- **Full document summaries**: 1024-2048 tokens

## 3. Metadata filtering to narrow search space

```python
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

filters = MetadataFilters(
    filters=[
        ExactMatchFilter(key="category", value="finance"),
        ExactMatchFilter(key="year", value="2024"),
    ]
)
retriever = index.as_retriever(filters=filters, similarity_top_k=5)
```

## 4. Re-ranking to improve precision

Fetch more candidates than you need, then rerank to return the most relevant.

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3,
)
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)
```

## 5. Caching embeddings

Avoid re-computing embeddings for unchanged documents.

```python
from llama_index.core import StorageContext, load_index_from_storage
from pathlib import Path

PERSIST_DIR = Path("./storage")

if PERSIST_DIR.exists():
    storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
    index = load_index_from_storage(storage_context)
else:
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
```

## 6. Async retrieval for throughput

```python
async def retrieve_all(queries: list[str]) -> list:
    retriever = index.as_retriever(similarity_top_k=5)
    import asyncio
    results = await asyncio.gather(
        *[retriever.aretrieve(q) for q in queries]
    )
    return results
```

## 7. Optimization checklist

- [ ] Chunk size tuned to query type
- [ ] Metadata on documents for pre-filtering
- [ ] Re-ranker applied for high-precision use cases
- [ ] Embeddings persisted and reloaded on restart
- [ ] `similarity_top_k` set to retrieve 2-3x more than the final `top_n`
- [ ] Async retrieval used for batch queries
