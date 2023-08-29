import random
import warnings
from typing import Tuple, Dict, List

from fastrepl.utils import prompt
from fastrepl.llm import completion, SUPPORTED_MODELS
from fastrepl.eval.base import BaseEvalWithoutReference
from fastrepl.eval.model.utils import mapping_from_labels


@prompt
def system_prompt(context, labels, label_keys):
    """You are master of classification who can classify any text according to the user's instructions.
    If user gave you the text, do step by step thinking first, and classify it.

    When do step-by-step thinking(less than 30 words), you must consider the following:
    {{context}}

    These are the labels you can use:
    {{labels}}

    For classification, only output one of these label keys:
    {{label_keys}}

    When responding, strictly follow this format:
    ### Thoghts
    <STEP_BY_STEP_THOUGHTS>

    ### Label
    <LABEL>"""


@prompt
def final_message_prompt(sample, context=""):
    """{% if context != '' %}
    Info about the text: {{ context }}
    {% endif %}
    Text to think about: {{ sample }}"""


class LLMChainOfThoughtClassifier(BaseEvalWithoutReference):
    __slots__ = ("model", "references", "rg", "system_msg")

    def __init__(
        self,
        context: str,
        labels: Dict[str, str],
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
        rg=random.Random(42),
        references: List[Tuple[str, str]] = [],
    ) -> None:
        self.context = context
        self.labels = labels
        self.model = model
        self.rg = rg
        self.references = references

    def _shuffle(self):
        mapping = mapping_from_labels(self.labels, rg=self.rg)
        references = self.rg.sample(self.references, len(self.references))
        return mapping, references

    def compute(self, sample: str, context="") -> str:
        mapping, references = self._shuffle()

        instruction = system_prompt(
            context=self.context,
            labels="\n".join(f"{m.token}: {m.description}" for m in mapping),
            label_keys=", ".join(m.token for m in mapping),
        )

        messages = [{"role": "system", "content": instruction}]
        for input, output in references:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})
        messages.append(
            {"role": "user", "content": final_message_prompt(sample, context)}
        )

        # fmt: off
        result = completion(
            model=self.model,
            messages=messages,
        )["choices"][0]["message"]["content"]
        # fmt: on

        for m in mapping:
            if m.token == result[-1]:
                return m.label

        warnings.warn(f"classification result not in mapping: {result!r}")
        return "UNKNOWN"

    def is_interactive(self) -> bool:
        return False
