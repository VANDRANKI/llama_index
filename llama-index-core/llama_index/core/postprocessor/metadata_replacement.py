"""Metadata replacement postprocessor."""

from typing import List, Optional

from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle


class MetadataReplacementPostProcessor(BaseNodePostprocessor):
    """
    Replace each node's content with the value of a given metadata key.

    This is useful for retrieval strategies where a node is embedded/indexed
    using one representation (e.g. a summary), but the metadata stores a
    richer or differently-formatted representation (e.g. the original
    windowed text) that should be surfaced to the LLM at query time. If the
    target metadata key is not present on a node, the node's existing
    content is left unchanged.

    Args:
        target_metadata_key (str): The metadata key whose value should
            replace the node's content.

    """

    target_metadata_key: str = Field(
        description="Target metadata key to replace node content with."
    )

    def __init__(self, target_metadata_key: str) -> None:
        super().__init__(target_metadata_key=target_metadata_key)

    @classmethod
    def class_name(cls) -> str:
        return "MetadataReplacementPostProcessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        for n in nodes:
            n.node.set_content(
                n.node.metadata.get(
                    self.target_metadata_key,
                    n.node.get_content(metadata_mode=MetadataMode.NONE),
                )
            )

        return nodes
