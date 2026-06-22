"""Base query engine."""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.prompts.mixin import PromptDictType, PromptMixin
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.instrumentation.events.query import (
    QueryEndEvent,
    QueryStartEvent,
)
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)
logger = logging.getLogger(__name__)


class BaseQueryEngine(PromptMixin, DispatcherSpanMixin):
    """Abstract base class for all LlamaIndex query engines.

    A query engine accepts a natural-language query string (or a pre-built
    ``QueryBundle``) and returns a synthesised response backed by one or more
    retrieved source nodes.  Concrete implementations must override
    ``_query`` and ``_aquery``; all other methods have default
    implementations that raise ``NotImplementedError``.

    Args:
        callback_manager (Optional[CallbackManager]): Callback manager used
            to emit trace events during query execution.  If ``None``, an
            empty ``CallbackManager`` is created automatically.

    Example:
        >>> from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        >>> documents = SimpleDirectoryReader("data").load_data()
        >>> index = VectorStoreIndex.from_documents(documents)
        >>> query_engine = index.as_query_engine()
        >>> response = query_engine.query("What did the author do growing up?")
        >>> print(response)
    """

    def __init__(
        self,
        callback_manager: Optional[CallbackManager],
    ) -> None:
        self.callback_manager = callback_manager or CallbackManager([])

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    @dispatcher.span
    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        """Execute a synchronous query against the index.

        Accepts either a plain query string or a pre-built ``QueryBundle``.
        The string is automatically wrapped in a ``QueryBundle`` before being
        dispatched to the underlying ``_query`` implementation.  Query start
        and end events are emitted on the dispatcher so that any attached
        instrumentation (e.g. tracing, logging) can observe the full
        lifecycle of the request.

        Args:
            str_or_query_bundle (QueryType): The query to execute.  Can be a
                plain ``str`` or an already-constructed ``QueryBundle``.

        Returns:
            RESPONSE_TYPE: The synthesised response, typically a
                ``Response``, ``StreamingResponse``, or
                ``AsyncStreamingResponse`` object depending on the concrete
                engine implementation.

        Example:
            >>> response = query_engine.query("What is the capital of France?")
            >>> print(response.response)
            Paris
        """
        dispatcher.event(QueryStartEvent(query=str_or_query_bundle))
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            query_result = self._query(str_or_query_bundle)
        dispatcher.event(
            QueryEndEvent(query=str_or_query_bundle, response=query_result)
        )
        return query_result

    @dispatcher.span
    async def aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        """Execute an asynchronous query against the index.

        The async counterpart to :meth:`query`.  Accepts either a plain query
        string or a pre-built ``QueryBundle`` and delegates to the concrete
        ``_aquery`` implementation.  Query start and end events are emitted
        on the dispatcher so that instrumentation can observe the full
        request lifecycle.

        Args:
            str_or_query_bundle (QueryType): The query to execute.  Can be a
                plain ``str`` or an already-constructed ``QueryBundle``.

        Returns:
            RESPONSE_TYPE: The synthesised response, typically a
                ``Response``, ``StreamingResponse``, or
                ``AsyncStreamingResponse`` object depending on the concrete
                engine implementation.

        Example:
            >>> import asyncio
            >>> response = asyncio.run(
            ...     query_engine.aquery("What is the capital of France?")
            ... )
            >>> print(response.response)
            Paris
        """
        dispatcher.event(QueryStartEvent(query=str_or_query_bundle))
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            query_result = await self._aquery(str_or_query_bundle)
        dispatcher.event(
            QueryEndEvent(query=str_or_query_bundle, response=query_result)
        )
        return query_result

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve source nodes for a query without synthesising a response.

        Args:
            query_bundle (QueryBundle): The query bundle to retrieve nodes for.

        Returns:
            List[NodeWithScore]: Ranked list of nodes with their relevance
                scores.

        Raises:
            NotImplementedError: Most query engines do not expose retrieval
                separately; use :meth:`query` instead.
        """
        raise NotImplementedError(
            "This query engine does not support retrieve, use query directly"
        )

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        """Synthesize a response from already-retrieved nodes.

        Args:
            query_bundle (QueryBundle): The original query.
            nodes (List[NodeWithScore]): Retrieved nodes to synthesise from.
            additional_source_nodes (Optional[Sequence[NodeWithScore]]):
                Extra source nodes to include in the response metadata but
                not necessarily in the synthesis context.

        Returns:
            RESPONSE_TYPE: The synthesised response.

        Raises:
            NotImplementedError: Most query engines do not expose synthesis
                separately; use :meth:`query` instead.
        """
        raise NotImplementedError(
            "This query engine does not support synthesize, use query directly"
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        """Asynchronously synthesize a response from already-retrieved nodes.

        The async counterpart to :meth:`synthesize`.

        Args:
            query_bundle (QueryBundle): The original query.
            nodes (List[NodeWithScore]): Retrieved nodes to synthesise from.
            additional_source_nodes (Optional[Sequence[NodeWithScore]]):
                Extra source nodes to include in the response metadata.

        Returns:
            RESPONSE_TYPE: The synthesised response.

        Raises:
            NotImplementedError: Most query engines do not expose async
                synthesis separately; use :meth:`aquery` instead.
        """
        raise NotImplementedError(
            "This query engine does not support asynthesize, use aquery directly"
        )

    @abstractmethod
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Subclass-specific synchronous query implementation.

        Args:
            query_bundle (QueryBundle): The normalised query bundle.

        Returns:
            RESPONSE_TYPE: The synthesised response.
        """

    @abstractmethod
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Subclass-specific asynchronous query implementation.

        Args:
            query_bundle (QueryBundle): The normalised query bundle.

        Returns:
            RESPONSE_TYPE: The synthesised response.
        """
