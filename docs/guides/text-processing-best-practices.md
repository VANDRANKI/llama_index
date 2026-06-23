# Text Processing Best Practices in LlamaIndex

This guide covers patterns for effective text chunking, cleaning, and metadata extraction when building RAG pipelines.

## Choosing a Text Splitter

LlamaIndex offers several splitters. Choose based on your data:

| Splitter | Best For |
|---|---|
| `SentenceSplitter` | General prose, articles |
| `CodeSplitter` | Source code files |
| `TokenTextSplitter` | Token-budget-aware chunking |
| `SemanticSplitterNodeParser` | Semantic coherence (requires embed model) |

## Configuring Chunk Size

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=512,        # tokens per chunk
    chunk_overlap=64,      # overlap between adjacent chunks
    paragraph_separator="\n\n",
)
```

**Rule of thumb**: chunk_size should be ~25% of your model's context window so retrieved chunks leave room for the prompt.

## Adding Metadata to Nodes

Metadata improves retrieval precision and citation quality:

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from typing import Any


def enrich_document_metadata(doc: Document, source_url: str) -> Document:
    """Attach source URL and word count to a document before indexing.

    Args:
        doc: The document to enrich.
        source_url: Canonical URL of the source page.

    Returns:
        The same document object with updated metadata.
    """
    word_count = len(doc.text.split())
    doc.metadata.update({
        "source_url": source_url,
        "word_count": word_count,
        "char_count": len(doc.text),
    })
    return doc
```

## Excluding Metadata from Embeddings

Some metadata should influence ranking but not the embedding itself:

```python
doc.excluded_embed_metadata_keys = ["file_path", "creation_date"]
doc.excluded_llm_metadata_keys = ["file_size"]
```

## Cleaning Text Before Indexing

```python
import re


def clean_document_text(text: str) -> str:
    """Remove common document noise before indexing.

    Args:
        text: Raw document text.

    Returns:
        Cleaned text with normalized whitespace and removed boilerplate.
    """
    # Collapse excessive whitespace
    text = re.sub(r"\s{3,}", "\n\n", text)
    # Remove page numbers like "Page 12 of 45"
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    # Strip leading/trailing whitespace
    return text.strip()
```

## Streaming Ingestion for Large Datasets

For large corpora, use the ingestion pipeline with async support:

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import resolve_embed_model

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512),
        resolve_embed_model("default"),
    ]
)

# Async ingestion
import asyncio
nodes = asyncio.run(pipeline.arun(documents=documents))
```

## Best Practices

1. **Preserve document boundaries** — don't merge chunks across separate source files.
2. **Tune overlap based on query type** — higher overlap (128–256 tokens) for factual Q&A, lower (32–64) for summarization.
3. **Index metadata separately** — use `KeywordTableIndex` alongside `VectorStoreIndex` for hybrid retrieval.
4. **Validate node counts** — log how many nodes each document produces to catch misconfiguration.
5. **Cache embeddings** — use `IngestionCache` to avoid re-embedding unchanged documents.
