"""Global constants used throughout the LlamaIndex core library.

This module centralises every magic number and sentinel string that would
otherwise be scattered across the codebase.  Import from here rather than
hardcoding values inline so that a single change is reflected everywhere.

Categories
----------
LLM / generation defaults
    Temperature, context window size, and output-token defaults that are
    used when no provider-specific value is available.

Chunking defaults
    Default chunk size and overlap for text splitters.

Retrieval defaults
    Top-k values for vector and image similarity search.

Embedding defaults
    Default dimensionality (text-embedding-ada-002).

Legacy provider context windows
    Hard-coded context windows for providers that predate the dynamic
    metadata system (Cohere, AI21).

Serialization keys
    Sentinel strings used when persisting index data structures to disk
    or a document store.

LlamaCloud constants
    Default pipeline / project names and API base URLs for LlamaCloud.
"""

# ---------------------------------------------------------------------------
# LLM / generation defaults
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE = 0.1
DEFAULT_CONTEXT_WINDOW = 3900  # tokens
DEFAULT_NUM_OUTPUTS = 256  # tokens
DEFAULT_NUM_INPUT_FILES = 10  # files

# ---------------------------------------------------------------------------
# Embedding defaults
# ---------------------------------------------------------------------------

DEFAULT_EMBED_BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# Chunking / text-splitting defaults
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 1024  # tokens
DEFAULT_CHUNK_OVERLAP = 20  # tokens

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------

DEFAULT_SIMILARITY_TOP_K = 2
DEFAULT_IMAGE_SIMILARITY_TOP_K = 2

# NOTE: for text-embedding-ada-002
DEFAULT_EMBEDDING_DIM = 1536

# ---------------------------------------------------------------------------
# Legacy provider context windows
# ---------------------------------------------------------------------------

# context window size for llm predictor
COHERE_CONTEXT_WINDOW = 2048
AI21_J2_CONTEXT_WINDOW = 8192

# ---------------------------------------------------------------------------
# Serialization / persistence keys
# ---------------------------------------------------------------------------

TYPE_KEY = "__type__"
DATA_KEY = "__data__"
VECTOR_STORE_KEY = "vector_store"
IMAGE_STORE_KEY = "image_store"
GRAPH_STORE_KEY = "graph_store"
INDEX_STORE_KEY = "index_store"
DOC_STORE_KEY = "doc_store"
PG_STORE_KEY = "property_graph_store"

# ---------------------------------------------------------------------------
# LlamaCloud constants
# ---------------------------------------------------------------------------

DEFAULT_PIPELINE_NAME = "default"
DEFAULT_PROJECT_NAME = "Default"
DEFAULT_BASE_URL = "https://api.cloud.llamaindex.ai"
DEFAULT_APP_URL = "https://cloud.llamaindex.ai"
