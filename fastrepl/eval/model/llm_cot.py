import random
import itertools
from typing import Optional, Tuple, Dict, List

from fastrepl.utils import prompt
from fastrepl.llm import completion, SUPPORTED_MODELS
from fastrepl.eval.base import BaseEvalNode


class LLMChainOfThought(BaseEvalNode):
    def __init__(
        self,
        context: str,
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
    ) -> None:
        self.global_context = context
        self.model = model
        self.rg = random.Random(42)
        self.references: List[Tuple[str, str]] = []

    def system_message(
        self, sample: str, global_context: str, local_context: Optional[str]
    ) -> Dict[str, str]:
        @prompt
        def p(context):
            """If user gave you the text, do step-by-step thinking about it within 3 sentences.

            When doing step-by-step thinking, you must consider the following:
            {{ context }}"""

        return {"role": "system", "content": p(global_context)}

    def final_message(
        self, sample: str, global_context: str, local_context: Optional[str]
    ) -> Dict[str, str]:
        @prompt
        def p(sample, context):
            """{% if context is not none %}
            Information or thought about the text: {{ context }}
            {% endif %}
            Text to think about: {{ sample }}"""

        return {"role": "user", "content": p(sample, local_context)}

    def compute(self, sample: str, context: Optional[str] = None) -> str:
        system_message = self.system_message(sample, self.global_context, context)
        final_message = self.final_message(sample, self.global_context, context)

        references = self.rg.sample(self.references, len(self.references))
        reference_messages = list(
            itertools.chain.from_iterable(
                (
                    {"role": "user", "content": input},
                    {"role": "assistant", "content": output},
                )
                for input, output in references
            )
        )

        messages = [system_message, *reference_messages, final_message]

        # fmt: off
        result = completion(
            model=self.model,
            messages=messages,
        )["choices"][0]["message"]["content"]
        # fmt: on

        return result
