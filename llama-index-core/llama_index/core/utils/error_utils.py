"""Error handling utilities for LlamaIndex core operations.

Provides consistent error wrapping and diagnostic helpers used across
index operations, query engines, and embedding pipelines.
"""

from __future__ import annotations

from typing import Any, Optional, Type


class LlamaIndexOperationError(Exception):
    """Base exception for operation-level errors in LlamaIndex.

    Wraps lower-level exceptions with context about which operation failed
    and why, making it easier to debug multi-step index pipelines.

    Attributes:
        operation: Human-readable description of the failed operation.
        cause: The original exception that triggered this error.
    """

    def __init__(
        self,
        operation: str,
        cause: Optional[BaseException] = None,
        detail: Optional[str] = None,
    ) -> None:
        self.operation = operation
        self.cause = cause
        parts = [f"Failed to {operation}"]
        if detail:
            parts.append(detail)
        if cause:
            parts.append(f"Caused by: {type(cause).__name__}: {cause}")
        super().__init__(" — ".join(parts))


def require_non_empty(
    value: Any,
    name: str,
    operation: str,
) -> Any:
    """Assert that *value* is non-empty, raising a descriptive error if not.

    Args:
        value: The value to check. Supports strings, lists, dicts, and any
            type with a truthiness check.
        name: The parameter name for the error message.
        operation: Description of the operation being performed.

    Returns:
        *value* unchanged.

    Raises:
        LlamaIndexOperationError: If *value* is falsy.
    """
    if not value:
        raise LlamaIndexOperationError(
            operation,
            detail=f"'{name}' must not be empty",
        )
    return value


def wrap_embedding_error(func: "Any") -> "Any":
    """Decorator that wraps embedding errors with a descriptive message.

    Args:
        func: The function to wrap. Should be an embedding call that may raise
            provider-specific exceptions.

    Returns:
        A wrapped version of *func* that raises ``LlamaIndexOperationError``
        on any exception.
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except LlamaIndexOperationError:
            raise
        except Exception as exc:
            raise LlamaIndexOperationError(
                "generate embeddings",
                cause=exc,
            ) from exc

    return wrapper
