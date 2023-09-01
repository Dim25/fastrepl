import random
from typing import Tuple, List, Dict

from fastrepl.utils import prompt
from fastrepl.llm import completion, SUPPORTED_MODELS
from fastrepl.eval.base import BaseEvalWithoutReference


@prompt
def system_prompt(labels, context):
    """If user gave you the text, do step-by-step thinking that is needed to classify the given text.
    Use less than 50 words and maximun 3 steps.

    These are the labels that will be used later to classify the text:
    {{labels}}

    When do step-by-step thinking, you must consider the following:
    {{context}}"""


@prompt
def final_message_prompt(sample, context=""):
    """{% if context != '' %}
    Info about the text: {{ context }}
    {% endif %}
    Text to think about: {{ sample }}"""


class LLMChainOfThought(BaseEvalWithoutReference):
    def __init__(
        self,
        context: str,
        labels: Dict[str, str],
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
        rg=random.Random(42),
        references: List[Tuple[str, str]] = [],
    ) -> None:
        self.global_context = context
        self.labels = labels
        self.model = model
        self.rg = rg
        self.references = references

    def _shuffle(self):
        references = self.rg.sample(self.references, len(self.references))
        return references

    def compute(self, prediction: str, context="") -> str:
        references = self._shuffle()

        instruction = system_prompt(
            context=self.global_context,
            labels="\n".join(f"{k}: {v}" for k, v in self.labels.items()),
        )

        messages = [{"role": "system", "content": instruction}]
        for input, output in references:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})
        messages.append(
            {"role": "user", "content": final_message_prompt(prediction, context)}
        )

        # fmt: off
        result = completion(
            model=self.model,
            messages=messages,
        )["choices"][0]["message"]["content"]
        # fmt: on

        return result

    def is_interactive(self) -> bool:
        return False
