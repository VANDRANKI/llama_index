"""Abstract base class for all query engines in LlamaIndex.

A query engine accepts a natural-language question (or a :class:`QueryBundle`)
and returns a synthesised response backed by an underlying index or retrieval
mechanism.  Sub-classes implement :meth:`_query` and :meth:`_aquery` and
optionally override :meth:`retrieve` / :meth:`synthesize` when those stages
are separable.
"""

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
    """Abstract base class for LlamaIndex query engines.

    A query engine wraps an index (or any retrieval + synthesis pipeline) and
    exposes a single :meth:`query` / :meth:`aquery` interface.  The class
    handles callback tracing and instrumentation; sub-classes only need to
    implement the private :meth:`_query` and :meth:`_aquery` methods.

    Args:
        callback_manager: Optional :class:`CallbackManager` used to emit
            trace events.  A new empty manager is created when ``None``
            is passed.

    Example::

        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        response = query_engine.query("What is the capital of France?")
        print(response)

    """

    def __init__(
        self,
        callback_manager: Optional[CallbackManager],
    ) -> None:
        self.callback_manager = callback_manager or CallbackManager([])

    def _get_prompts(self) -> Dict[str, Any]:
        """Return a mapping of prompt keys to their current prompt templates.

        Sub-classes that expose configurable prompts should override this
        method and return each prompt keyed by a human-readable name.
        The default implementation returns an empty dict.
        """
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Apply a mapping of updated prompts returned by :meth:`_get_prompts`.

        Sub-classes that expose configurable prompts should override this
        method to update their internal prompt references.  The default
        implementation is a no-op.

        Args:
            prompts: A dict mapping prompt keys (as returned by
                :meth:`_get_prompts`) to new prompt values.
        """

    @dispatcher.span
    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        """Query the engine with a string or :class:`QueryBundle` (synchronous).

        Wraps :meth:`_query` with callback tracing and instrumentation
        events.  Prefer :meth:`aquery` in async contexts to avoid blocking
        the event loop.

        Args:
            str_or_query_bundle: Either a plain query string or a
                :class:`QueryBundle` containing the query string and
                optional image/embedding overrides.

        Returns:
            A :data:`RESPONSE_TYPE` instance (typically
            :class:`~llama_index.core.base.response.schema.Response` or
            :class:`~llama_index.core.base.response.schema.StreamingResponse`).

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
        """Query the engine with a string or :class:`QueryBundle` (async).

        Async counterpart of :meth:`query`.  Wraps :meth:`_aquery` with
        callback tracing and instrumentation events.

        Args:
            str_or_query_bundle: Either a plain query string or a
                :class:`QueryBundle` containing the query string and
                optional image/embedding overrides.

        Returns:
            A :data:`RESPONSE_TYPE` instance.

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
        """Retrieve source nodes for *query_bundle* without synthesising a response.

        Not all query engines separate retrieval from synthesis.  The default
        implementation raises :exc:`NotImplementedError`; engines that support
        this stage (e.g. :class:`RetrieverQueryEngine`) override it.

        Args:
            query_bundle: The query to retrieve nodes for.

        Returns:
            A list of :class:`NodeWithScore` objects.

        Raises:
            NotImplementedError: If the engine does not support standalone
                retrieval.

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
        """Synthesise a response from pre-retrieved *nodes* (synchronous).

        Allows callers to decouple retrieval from synthesis, for example to
        inject custom re-ranking between the two stages.  The default
        implementation raises :exc:`NotImplementedError`.

        Args:
            query_bundle: The original query.
            nodes: Retrieved nodes to synthesise over.
            additional_source_nodes: Optional extra nodes included as
                sources in the response metadata but not synthesised over.

        Returns:
            A :data:`RESPONSE_TYPE` instance.

        Raises:
            NotImplementedError: If the engine does not support standalone
                synthesis.

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
        """Synthesise a response from pre-retrieved *nodes* (async).

        Async counterpart of :meth:`synthesize`.  The default implementation
        raises :exc:`NotImplementedError`.

        Args:
            query_bundle: The original query.
            nodes: Retrieved nodes to synthesise over.
            additional_source_nodes: Optional extra nodes included as
                sources in the response metadata but not synthesised over.

        Returns:
            A :data:`RESPONSE_TYPE` instance.

        Raises:
            NotImplementedError: If the engine does not support standalone
                async synthesis.

        """
        raise NotImplementedError(
            "This query engine does not support asynthesize, use aquery directly"
        )

    @abstractmethod
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Execute the query and return a synthesised response (synchronous).

        This is the core method that sub-classes must implement.  It receives
        a :class:`QueryBundle` and should return a fully synthesised response.
        Callback tracing is handled by the public :meth:`query` wrapper.

        Args:
            query_bundle: The parsed query object.

        Returns:
            A :data:`RESPONSE_TYPE` instance.

        """

    @abstractmethod
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Execute the query and return a synthesised response (async).

        Async counterpart of :meth:`_query`.  Sub-classes must implement
        this method.  Callback tracing is handled by the public
        :meth:`aquery` wrapper.

        Args:
            query_bundle: The parsed query object.

        Returns:
            A :data:`RESPONSE_TYPE` instance.

        """
