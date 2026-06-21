"""Global settings singleton for the LlamaIndex library.

This module exposes a single :data:`Settings` instance (of :class:`_Settings`)
that acts as the central configuration object for LlamaIndex.  Users can
override any component globally, and all LlamaIndex internals will pick up
the change automatically.

Typical usage::

    from llama_index.core import Settings

    # Override the default LLM
    from llama_index.llms.openai import OpenAI
    Settings.llm = OpenAI(model="gpt-4o")

    # Override the embedding model
    from llama_index.embeddings.openai import OpenAIEmbedding
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Tune chunking behaviour
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional


from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import BaseCallbackHandler, CallbackManager
from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms import LLM
from llama_index.core.llms.utils import LLMType, resolve_llm
from llama_index.core.node_parser import NodeParser, SentenceSplitter
from llama_index.core.schema import TransformComponent
from llama_index.core.types import PydanticProgramMode
from llama_index.core.utils import get_tokenizer, set_global_tokenizer


@dataclass
class _Settings:
    """Lazy-initialised global settings for LlamaIndex.

    All attributes are initialised on first access rather than at import
    time so that users can import the module before setting API keys or
    installing optional dependencies.

    Do not instantiate this class directly.  Use the module-level
    :data:`Settings` singleton instead.

    Attributes:
        llm: The language model used for completions and chat.  Defaults
            to the provider resolved by ``resolve_llm("default")``.
        embed_model: The embedding model used to vectorise text.  Defaults
            to the provider resolved by ``resolve_embed_model("default")``.
        callback_manager: The :class:`CallbackManager` injected into the
            LLM, embedding model, and node parser on every access.
        tokenizer: A callable that maps a string to a list of tokens.  Used
            internally to count tokens without making API calls.
        node_parser: The node parser used to split documents into nodes.
            Defaults to :class:`SentenceSplitter`.
        prompt_helper: Controls context-window management and prompt
            truncation.  Built automatically from the LLM metadata when
            not set explicitly.
        transformations: The transformation pipeline applied during
            ingestion.  Defaults to ``[node_parser]``.
    """

    # lazy initialization
    _llm: Optional[LLM] = None
    _embed_model: Optional[BaseEmbedding] = None
    _callback_manager: Optional[CallbackManager] = None
    _tokenizer: Optional[Callable[[str], List[Any]]] = None
    _node_parser: Optional[NodeParser] = None
    _prompt_helper: Optional[PromptHelper] = None
    _transformations: Optional[List[TransformComponent]] = None

    # ---- LLM ----

    @property
    def llm(self) -> LLM:
        """Get the LLM."""
        if self._llm is None:
            self._llm = resolve_llm("default")

        if self._callback_manager is not None:
            self._llm.callback_manager = self._callback_manager

        return self._llm

    @llm.setter
    def llm(self, llm: LLMType) -> None:
        """Set the LLM."""
        self._llm = resolve_llm(llm)

    @property
    def pydantic_program_mode(self) -> PydanticProgramMode:
        """Get the pydantic program mode."""
        return self.llm.pydantic_program_mode

    @pydantic_program_mode.setter
    def pydantic_program_mode(self, pydantic_program_mode: PydanticProgramMode) -> None:
        """Set the pydantic program mode."""
        self.llm.pydantic_program_mode = pydantic_program_mode

    # ---- Embedding ----

    @property
    def embed_model(self) -> BaseEmbedding:
        """Get the embedding model."""
        if self._embed_model is None:
            self._embed_model = resolve_embed_model("default")

        if self._callback_manager is not None:
            self._embed_model.callback_manager = self._callback_manager

        return self._embed_model

    @embed_model.setter
    def embed_model(self, embed_model: EmbedType) -> None:
        """Set the embedding model."""
        self._embed_model = resolve_embed_model(embed_model)

    # ---- Callbacks ----

    @property
    def global_handler(self) -> Optional[BaseCallbackHandler]:
        """Get the global handler."""
        import llama_index.core

        # TODO: deprecated?
        return llama_index.core.global_handler

    @global_handler.setter
    def global_handler(self, eval_mode: str, **eval_params: Any) -> None:
        """Set the global handler."""
        from llama_index.core import set_global_handler

        # TODO: deprecated?
        set_global_handler(eval_mode, **eval_params)

    @property
    def callback_manager(self) -> CallbackManager:
        """Get the callback manager."""
        if self._callback_manager is None:
            self._callback_manager = CallbackManager()
        return self._callback_manager

    @callback_manager.setter
    def callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set the callback manager."""
        self._callback_manager = callback_manager

    # ---- Tokenizer ----

    @property
    def tokenizer(self) -> Callable[[str], List[Any]]:
        """Get the tokenizer."""
        import llama_index.core

        if llama_index.core.global_tokenizer is None:
            return get_tokenizer()

        # TODO: deprecated?
        return llama_index.core.global_tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Callable[[str], List[Any]]) -> None:
        """Set the tokenizer.

        If a HuggingFace ``PreTrainedTokenizerBase`` is supplied, it is
        automatically wrapped so that it returns token IDs without special
        tokens, matching the interface expected by LlamaIndex internals.
        """
        try:
            from transformers import PreTrainedTokenizerBase  # pants: no-infer-dep

            if isinstance(tokenizer, PreTrainedTokenizerBase):
                from functools import partial

                tokenizer = partial(tokenizer.encode, add_special_tokens=False)
        except ImportError:
            pass

        # TODO: deprecated?
        set_global_tokenizer(tokenizer)

    # ---- Node parser ----

    @property
    def node_parser(self) -> NodeParser:
        """Get the node parser."""
        if self._node_parser is None:
            self._node_parser = SentenceSplitter()

        if self._callback_manager is not None:
            self._node_parser.callback_manager = self._callback_manager

        return self._node_parser

    @node_parser.setter
    def node_parser(self, node_parser: NodeParser) -> None:
        """Set the node parser."""
        self._node_parser = node_parser

    @property
    def chunk_size(self) -> int:
        """Get the chunk size from the configured node parser.

        Raises:
            ValueError: If the current node parser does not expose a
                ``chunk_size`` attribute.
        """
        if hasattr(self.node_parser, "chunk_size"):
            return self.node_parser.chunk_size
        else:
            raise ValueError("Configured node parser does not have chunk size.")

    @chunk_size.setter
    def chunk_size(self, chunk_size: int) -> None:
        """Set the chunk size on the configured node parser.

        Raises:
            ValueError: If the current node parser does not expose a
                ``chunk_size`` attribute.
        """
        if hasattr(self.node_parser, "chunk_size"):
            self.node_parser.chunk_size = chunk_size
        else:
            raise ValueError("Configured node parser does not have chunk size.")

    @property
    def chunk_overlap(self) -> int:
        """Get the chunk overlap from the configured node parser.

        Raises:
            ValueError: If the current node parser does not expose a
                ``chunk_overlap`` attribute.
        """
        if hasattr(self.node_parser, "chunk_overlap"):
            return self.node_parser.chunk_overlap
        else:
            raise ValueError("Configured node parser does not have chunk overlap.")

    @chunk_overlap.setter
    def chunk_overlap(self, chunk_overlap: int) -> None:
        """Set the chunk overlap on the configured node parser.

        Raises:
            ValueError: If the current node parser does not expose a
                ``chunk_overlap`` attribute.
        """
        if hasattr(self.node_parser, "chunk_overlap"):
            self.node_parser.chunk_overlap = chunk_overlap
        else:
            raise ValueError("Configured node parser does not have chunk overlap.")

    # ---- Node parser alias ----

    @property
    def text_splitter(self) -> NodeParser:
        """Get the text splitter (alias for :attr:`node_parser`)."""
        return self.node_parser

    @text_splitter.setter
    def text_splitter(self, text_splitter: NodeParser) -> None:
        """Set the text splitter (alias for :attr:`node_parser`)."""
        self.node_parser = text_splitter

    @property
    def prompt_helper(self) -> PromptHelper:
        """Get the prompt helper.

        If no prompt helper has been set and an LLM is configured, the
        helper is constructed from the LLM's metadata.  Otherwise a
        default :class:`PromptHelper` is used.
        """
        if self._llm is not None and self._prompt_helper is None:
            self._prompt_helper = PromptHelper.from_llm_metadata(self._llm.metadata)
        elif self._prompt_helper is None:
            self._prompt_helper = PromptHelper()

        return self._prompt_helper

    @prompt_helper.setter
    def prompt_helper(self, prompt_helper: PromptHelper) -> None:
        """Set the prompt helper."""
        self._prompt_helper = prompt_helper

    @property
    def num_output(self) -> int:
        """Get the maximum number of output tokens from the prompt helper."""
        return self.prompt_helper.num_output

    @num_output.setter
    def num_output(self, num_output: int) -> None:
        """Set the maximum number of output tokens on the prompt helper."""
        self.prompt_helper.num_output = num_output

    @property
    def context_window(self) -> int:
        """Get the context window size from the prompt helper."""
        return self.prompt_helper.context_window

    @context_window.setter
    def context_window(self, context_window: int) -> None:
        """Set the context window size on the prompt helper."""
        self.prompt_helper.context_window = context_window

    # ---- Transformations ----

    @property
    def transformations(self) -> List[TransformComponent]:
        """Get the ingestion transformation pipeline.

        Defaults to ``[node_parser]`` when not explicitly configured.
        """
        if self._transformations is None:
            self._transformations = [self.node_parser]
        return self._transformations

    @transformations.setter
    def transformations(self, transformations: List[TransformComponent]) -> None:
        """Set the ingestion transformation pipeline."""
        self._transformations = transformations


# Singleton
Settings = _Settings()
