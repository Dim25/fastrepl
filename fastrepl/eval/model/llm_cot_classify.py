import random
import warnings
from typing import Tuple, Dict, List

from fastrepl.utils import prompt
from fastrepl.llm import completion, SUPPORTED_MODELS
from fastrepl.eval.base import BaseEval
from fastrepl.eval.model.utils import render_labels, mapping_from_labels


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


class LLMChainOfThoughtClassifier(BaseEval):
    __slots__ = ("model", "mapping", "references", "rg", "system_msg")

    def __init__(
        self,
        context: str,
        labels: Dict[str, str],
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
        rg=random.Random(42),
        references: List[Tuple[str, str]] = [],
    ) -> None:
        self.model = model
        self.mapping = mapping_from_labels(labels)
        self.references = references
        self.rg = rg
        self.system_msg = {
            "role": "system",
            "content": system_prompt(
                context=context,
                labels=render_labels(self.mapping),
                label_keys=", ".join(self.mapping.keys()),
            ),
        }

    def compute(self, sample: str, context="") -> str:
        references = self.rg.sample(self.references, len(self.references))

        messages = [self.system_msg]
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

        try:
            return self.mapping[result[-1]]
        except KeyError:
            warnings.warn(f"classification result not in mapping: {result!r}")
            return "UNKNOWN"

    def is_interactive(self) -> bool:
        return False
