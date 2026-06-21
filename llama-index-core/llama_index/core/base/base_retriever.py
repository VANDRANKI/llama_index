"""Abstract base class for all retrievers in LlamaIndex.

A retriever accepts a natural-language query and returns a ranked list of
:class:`~llama_index.core.schema.NodeWithScore` objects drawn from an
underlying index or document store.  Sub-classes implement :meth:`_retrieve`
and optionally :meth:`_aretrieve` for async operation.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixin,
    PromptMixinType,
)
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeWithScore,
    QueryBundle,
    QueryType,
    TextNode,
)
from llama_index.core.settings import Settings
from llama_index.core.utils import print_text
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


class BaseRetriever(PromptMixin, DispatcherSpanMixin):
    """Abstract base class for LlamaIndex retrievers.

    A retriever maps a query to a list of relevant
    :class:`~llama_index.core.schema.NodeWithScore` objects.  The class
    handles callback tracing, object-map lookups, and recursive retrieval
    (i.e. ``IndexNode`` objects that point to nested retrievers or query
    engines); sub-classes only need to implement :meth:`_retrieve`.

    Args:
        callback_manager: Optional :class:`CallbackManager` used to emit
            trace events.  Defaults to an empty manager.
        object_map: An optional mapping from ``index_id`` strings to
            arbitrary retrievable objects.  Used for recursive retrieval
            when an ``IndexNode`` does not carry its object inline.
        objects: An optional list of :class:`IndexNode` objects whose
            ``obj`` attributes are registered in *object_map* automatically.
            Mutually exclusive with supplying *object_map* directly.
        verbose: If ``True``, print diagnostic messages during recursive
            retrieval to aid debugging.

    """

    def __init__(
        self,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[Dict] = None,
        objects: Optional[List[IndexNode]] = None,
        verbose: bool = False,
    ) -> None:
        self.callback_manager = callback_manager or CallbackManager()

        if objects is not None:
            object_map = {obj.index_id: obj.obj for obj in objects}

        self.object_map = object_map or {}
        self._verbose = verbose

    def _check_callback_manager(self) -> None:
        """Ensure *callback_manager* is set, falling back to the global one.

        This guard is needed because some code paths instantiate retrievers
        without calling ``super().__init__``, leaving the attribute unset.
        """
        if not hasattr(self, "callback_manager"):
            self.callback_manager = Settings.callback_manager

    def _get_prompts(self) -> PromptDictType:
        """Return configurable prompt templates (empty by default)."""
        return {}

    def _get_prompt_modules(self) -> PromptMixinType:
        """Return sub-modules that expose their own prompts (empty by default)."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Apply updated prompts (no-op by default)."""

    def _retrieve_from_object(
        self,
        obj: Any,
        query_bundle: QueryBundle,
        score: float,
    ) -> List[NodeWithScore]:
        """Dispatch retrieval to a nested retrievable object.

        Handles four cases:

        * :class:`NodeWithScore` — returned as-is.
        * :class:`BaseNode` — wrapped in a :class:`NodeWithScore`.
        * :class:`BaseQueryEngine` — queried synchronously; the response
          string becomes a :class:`TextNode`.
        * :class:`BaseRetriever` — its :meth:`retrieve` method is called
          recursively.

        Args:
            obj: The object to retrieve from.
            query_bundle: The active query.
            score: Relevance score to assign when wrapping plain nodes.

        Returns:
            A list of :class:`NodeWithScore` instances.

        Raises:
            ValueError: If *obj* is not a recognised retrievable type.

        """
        if self._verbose:
            print_text(
                f"Retrieving from object {obj.__class__.__name__} with query {query_bundle.query_str}\n",
                color="llama_pink",
            )
        if isinstance(obj, NodeWithScore):
            return [obj]
        elif isinstance(obj, BaseNode):
            return [NodeWithScore(node=obj, score=score)]
        elif isinstance(obj, BaseQueryEngine):
            response = obj.query(query_bundle)
            return [
                NodeWithScore(
                    node=TextNode(text=str(response), metadata=response.metadata or {}),
                    score=score,
                )
            ]
        elif isinstance(obj, BaseRetriever):
            return obj.retrieve(query_bundle)
        else:
            raise ValueError(f"Object {obj} is not retrievable.")

    async def _aretrieve_from_object(
        self,
        obj: Any,
        query_bundle: QueryBundle,
        score: float,
    ) -> List[NodeWithScore]:
        """Async counterpart of :meth:`_retrieve_from_object`.

        Args:
            obj: The object to retrieve from.
            query_bundle: The active query.
            score: Relevance score to assign when wrapping plain nodes.

        Returns:
            A list of :class:`NodeWithScore` instances.

        Raises:
            ValueError: If *obj* is not a recognised retrievable type.

        """
        if isinstance(obj, NodeWithScore):
            return [obj]
        elif isinstance(obj, BaseNode):
            return [NodeWithScore(node=obj, score=score)]
        elif isinstance(obj, BaseQueryEngine):
            response = await obj.aquery(query_bundle)
            return [
                NodeWithScore(
                    node=TextNode(text=str(response), metadata=response.metadata or {}),
                    score=score,
                )
            ]
        elif isinstance(obj, BaseRetriever):
            return await obj.aretrieve(query_bundle)
        else:
            raise ValueError(f"Object {obj} is not retrievable.")

    def _handle_recursive_retrieval(
        self, query_bundle: QueryBundle, nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Expand ``IndexNode`` placeholders into their underlying nodes.

        For each node in *nodes*, if the node is an :class:`IndexNode`, this
        method looks up the corresponding object in :attr:`object_map` (or
        uses the inline ``obj`` attribute) and recursively retrieves from it.
        Plain :class:`TextNode` or :class:`ImageNode` objects are passed
        through unchanged.

        Duplicate nodes (same ``node_id``) are removed from the output while
        preserving insertion order.

        Args:
            query_bundle: The active query, forwarded to nested retrieval.
            nodes: Raw nodes returned by :meth:`_retrieve`.

        Returns:
            A de-duplicated list of :class:`NodeWithScore` instances.

        """
        retrieved_nodes: List[NodeWithScore] = []
        for n in nodes:
            node = n.node
            score = n.score or 1.0
            if isinstance(node, IndexNode):
                obj = node.obj or self.object_map.get(node.index_id, None)
                if obj is not None:
                    if self._verbose:
                        print_text(
                            f"Retrieval entering {node.index_id}: {obj.__class__.__name__}\n",
                            color="llama_turquoise",
                        )
                    retrieved_nodes.extend(
                        self._retrieve_from_object(
                            obj, query_bundle=query_bundle, score=score
                        )
                    )
                else:
                    retrieved_nodes.append(n)
            else:
                retrieved_nodes.append(n)

        seen = set()
        return [
            n
            for n in retrieved_nodes
            if not (
                n.node.node_id in seen or seen.add(n.node.node_id)  # type: ignore[func-returns-value]
            )
        ]

    async def _ahandle_recursive_retrieval(
        self, query_bundle: QueryBundle, nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Async counterpart of :meth:`_handle_recursive_retrieval`.

        Expands ``IndexNode`` placeholders using async dispatch and removes
        duplicate nodes from the result.

        Args:
            query_bundle: The active query, forwarded to nested retrieval.
            nodes: Raw nodes returned by :meth:`_aretrieve`.

        Returns:
            A de-duplicated list of :class:`NodeWithScore` instances.

        """
        retrieved_nodes: List[NodeWithScore] = []
        for n in nodes:
            node = n.node
            score = n.score or 1.0
            if isinstance(node, IndexNode):
                obj = node.obj or self.object_map.get(node.index_id, None)
                if obj is not None:
                    if self._verbose:
                        print_text(
                            f"Retrieval entering {node.index_id}: {obj.__class__.__name__}\n",
                            color="llama_turquoise",
                        )
                    # TODO: Add concurrent execution via `run_jobs()` ?
                    retrieved_nodes.extend(
                        await self._aretrieve_from_object(
                            obj, query_bundle=query_bundle, score=score
                        )
                    )
                else:
                    retrieved_nodes.append(n)
            else:
                retrieved_nodes.append(n)

        # remove any duplicates based on node_id
        seen = set()
        return [
            n
            for n in retrieved_nodes
            if not (
                n.node.node_id in seen or seen.add(n.node.node_id)  # type: ignore[func-returns-value]
            )
        ]

    @dispatcher.span
    def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        """Retrieve nodes given a query (synchronous).

        Wraps :meth:`_retrieve` with callback tracing, recursive
        ``IndexNode`` expansion, and instrumentation events.  Prefer
        :meth:`aretrieve` in async contexts.

        Args:
            str_or_query_bundle: Either a plain query string or a
                :class:`QueryBundle`.

        Returns:
            A list of :class:`NodeWithScore` objects ordered by relevance.

        """
        self._check_callback_manager()
        dispatcher.event(
            RetrievalStartEvent(
                str_or_query_bundle=str_or_query_bundle,
            )
        )
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self._retrieve(query_bundle)
                nodes = self._handle_recursive_retrieval(query_bundle, nodes)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatcher.event(
            RetrievalEndEvent(
                str_or_query_bundle=str_or_query_bundle,
                nodes=nodes,
            )
        )
        return nodes

    @dispatcher.span
    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        """Retrieve nodes given a query (async).

        Async counterpart of :meth:`retrieve`.  Wraps :meth:`_aretrieve`
        with callback tracing, recursive ``IndexNode`` expansion, and
        instrumentation events.

        Args:
            str_or_query_bundle: Either a plain query string or a
                :class:`QueryBundle`.

        Returns:
            A list of :class:`NodeWithScore` objects ordered by relevance.

        """
        self._check_callback_manager()

        dispatcher.event(
            RetrievalStartEvent(
                str_or_query_bundle=str_or_query_bundle,
            )
        )
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self._aretrieve(query_bundle=query_bundle)
                nodes = await self._ahandle_recursive_retrieval(
                    query_bundle=query_bundle, nodes=nodes
                )
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatcher.event(
            RetrievalEndEvent(
                str_or_query_bundle=str_or_query_bundle,
                nodes=nodes,
            )
        )
        return nodes

    @abstractmethod
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given a query (synchronous, to be implemented).

        Sub-classes must implement this method.  It should perform the
        actual index lookup and return scored nodes.  Callback tracing and
        recursive expansion are handled by the public :meth:`retrieve`
        wrapper.

        Args:
            query_bundle: The parsed query object.

        Returns:
            A list of :class:`NodeWithScore` objects.

        """

    # TODO: make this abstract
    # @abstractmethod
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given a query (async, to be implemented).

        The default implementation delegates to the synchronous
        :meth:`_retrieve`.  Sub-classes with a native async backend should
        override this method to avoid blocking the event loop.

        Args:
            query_bundle: The parsed query object.

        Returns:
            A list of :class:`NodeWithScore` objects.

        """
        return self._retrieve(query_bundle)
