from typing import Any

from llama_index.core.output_parsers.base import StructuredOutput
from llama_index.core.output_parsers.utils import parse_json_markdown
from llama_index.core.question_gen.types import SubQuestion
from llama_index.core.types import BaseOutputParser


class SubQuestionOutputParser(BaseOutputParser):
    """Parses an LLM's raw text output into a list of `SubQuestion` objects."""

    def parse(self, output: str) -> Any:
        """
        Parse the raw LLM output into a `StructuredOutput` of sub questions.

        Args:
            output: The raw text produced by the LLM, expected to contain a
                JSON array (or object with an `items` key) of sub-question
                dicts.

        Returns:
            A `StructuredOutput` whose `parsed_output` is the list of
            parsed `SubQuestion` instances.

        Raises:
            ValueError: If no valid JSON could be extracted from `output`.

        """
        json_dict = parse_json_markdown(output)
        if not json_dict:
            raise ValueError(f"No valid JSON found in output: {output}")

        # example code includes an 'items' key, which breaks
        # the parsing from open-source LLMs such as Zephyr.
        # This gets the actual subquestions and recommended tools directly
        if "items" in json_dict:
            json_dict = json_dict["items"]

        sub_questions = [SubQuestion.model_validate(item) for item in json_dict]
        return StructuredOutput(raw_output=output, parsed_output=sub_questions)

    def format(self, prompt_template: str) -> str:
        """Return the prompt template unmodified (no formatting instructions to inject)."""
        return prompt_template
