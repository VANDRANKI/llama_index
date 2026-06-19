"""Unit tests for llama_index.core.utils.error_utils."""

import pytest
from llama_index.core.utils.error_utils import (
    LlamaIndexOperationError,
    require_non_empty,
    wrap_embedding_error,
)


class TestLlamaIndexOperationError:
    def test_basic_message(self):
        err = LlamaIndexOperationError("index documents")
        assert "Failed to index documents" in str(err)

    def test_with_detail(self):
        err = LlamaIndexOperationError("query index", detail="no nodes found")
        assert "no nodes found" in str(err)

    def test_with_cause(self):
        cause = ValueError("connection refused")
        err = LlamaIndexOperationError("embed text", cause=cause)
        assert "ValueError" in str(err)
        assert "connection refused" in str(err)

    def test_stores_operation(self):
        err = LlamaIndexOperationError("store vectors")
        assert err.operation == "store vectors"

    def test_stores_cause(self):
        cause = RuntimeError("boom")
        err = LlamaIndexOperationError("do something", cause=cause)
        assert err.cause is cause


class TestRequireNonEmpty:
    def test_passes_through_non_empty_string(self):
        result = require_non_empty("hello", "text", "process text")
        assert result == "hello"

    def test_passes_through_non_empty_list(self):
        result = require_non_empty([1, 2], "nodes", "index nodes")
        assert result == [1, 2]

    def test_raises_on_empty_string(self):
        with pytest.raises(LlamaIndexOperationError) as exc_info:
            require_non_empty("", "query", "execute query")
        assert "query" in str(exc_info.value)

    def test_raises_on_empty_list(self):
        with pytest.raises(LlamaIndexOperationError):
            require_non_empty([], "nodes", "index nodes")

    def test_raises_on_none(self):
        with pytest.raises(LlamaIndexOperationError):
            require_non_empty(None, "embedding", "generate embedding")


class TestWrapEmbeddingError:
    def test_passes_through_on_success(self):
        @wrap_embedding_error
        def embed(text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

        assert embed("hello") == [0.1, 0.2, 0.3]

    def test_wraps_exception(self):
        @wrap_embedding_error
        def embed_fail(text: str) -> list[float]:
            raise ConnectionError("provider down")

        with pytest.raises(LlamaIndexOperationError) as exc_info:
            embed_fail("hello")
        assert "generate embeddings" in str(exc_info.value)
        assert isinstance(exc_info.value.cause, ConnectionError)

    def test_does_not_double_wrap(self):
        @wrap_embedding_error
        def already_wrapped(text: str) -> list[float]:
            raise LlamaIndexOperationError("inner op")

        with pytest.raises(LlamaIndexOperationError) as exc_info:
            already_wrapped("hello")
        assert exc_info.value.operation == "inner op"
