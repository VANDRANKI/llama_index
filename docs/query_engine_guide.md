# LlamaIndex Query Engine Selection Guide

This guide helps you choose the right query engine for your use case
and explains how to compose multiple retrievers for complex workloads.

## Query Engine Types

### 1. VectorStoreIndex (Most Common)

Best for: semantic similarity search over unstructured documents.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)

response = query_engine.query("What is the refund policy?")
print(response)
```

**When to use**: General Q&A, search over knowledge bases, customer support bots.
**Limitation**: Doesn't handle multi-document aggregation ("summarize all docs") well.

### 2. SummaryIndex (Aggregation Queries)

Best for: questions that require reading all or most documents.

```python
from llama_index.core import SummaryIndex

index = SummaryIndex.from_documents(documents)
# Use tree summarize for large document sets (avoids context overflow)
query_engine = index.as_query_engine(response_mode="tree_summarize")

response = query_engine.query("Summarize the key themes across all documents.")
```

**When to use**: Summarization, trend analysis across many docs, report generation.
**Limitation**: Slower and more expensive — reads all nodes.

### 3. RouterQueryEngine (Multi-Strategy)

Combines multiple engines and routes queries to the best one:

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

vector_engine = vector_index.as_query_engine(similarity_top_k=5)
summary_engine = summary_index.as_query_engine(response_mode="tree_summarize")

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_engine,
    description="Useful for specific factual questions about the documentation.",
)
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_engine,
    description="Useful for high-level summaries and trend analysis.",
)

router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[vector_tool, summary_tool],
)

response = router_engine.query("What is the refund policy?")     # routes to vector
response = router_engine.query("Summarize all the documents.")   # routes to summary
```

## Debugging Retrieval Quality

When responses are poor, inspect which nodes were retrieved and their scores:

```python
from llama_index.core import VectorStoreIndex, Response

query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact",
)

response = query_engine.query("What is the cancellation policy?")

# Inspect source nodes and relevance scores
for node in response.source_nodes:
    print(f"Score: {node.score:.3f}")
    print(f"File: {node.metadata.get('file_name', 'unknown')}")
    print(f"Text: {node.text[:200]}")
    print()
```

Low scores (< 0.5) or off-topic nodes indicate the chunk size or embedding
model needs tuning.

## Tuning Chunk Size

The default chunk size (1024 tokens) is too large for precise retrieval
and too small for context-heavy documents. Rule of thumb:

| Use Case              | chunk_size | chunk_overlap |
|-----------------------|-----------|---------------|
| Short FAQ answers     | 256       | 32            |
| Technical docs        | 512       | 64            |
| Legal/financial docs  | 1024      | 128           |
| Book-length content   | 2048      | 256           |

```python
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)

# Now build the index — new chunk settings will apply
index = VectorStoreIndex.from_documents(documents)
```

## Async Query Engines

For web servers, always use async to avoid blocking:

```python
query_engine = index.as_query_engine(use_async=True)

async def handle_question(question: str) -> str:
    response = await query_engine.aquery(question)
    return str(response)
```

## SimpleDirectoryReader and Hidden Files

`SimpleDirectoryReader` skips hidden files (names starting with `.`) by
default. To include them:

```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_dir="./docs",
    exclude_hidden=False,  # include .hidden_file.txt
    recursive=True,
)
documents = reader.load_data()
```
