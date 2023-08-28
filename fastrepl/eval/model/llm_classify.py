import random
from typing import Tuple, Dict, List

from fastrepl.llm import completion, SUPPORTED_MODELS
from fastrepl.eval.base import BaseEval
from fastrepl.eval.model.utils import (
    render_labels,
    logit_bias_from_labels,
    mapping_from_labels,
)

LLM_CLASSIFIER_SYSTEM_TPL = """You are master of classification who can classify any text according to the user's instructions.
{context}

These are the labels you can use:
{labels}

Only output one of these label keys:
{label_keys}"""


class LLMClassifier(BaseEval):
    __slots__ = ("model", "mapping", "rg", "references", "system_msg")

    def __init__(
        self,
        labels: Dict[str, str],
        context: str = "",
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
        rg=random.Random(42),
        references: List[Tuple[str, str]] = [],
    ) -> None:
        self.model = model
        self.mapping = mapping_from_labels(labels)
        self.rg = rg
        self.references = references
        self.system_msg = {
            "role": "system",
            "content": LLM_CLASSIFIER_SYSTEM_TPL.format(
                context=context,
                labels=render_labels(self.mapping),
                label_keys=self.mapping.keys(),
            ),
        }

    def compute(self, sample: str, context="") -> str:
        references = self.rg.sample(self.references, len(self.references))

        messages = [self.system_msg]
        for input, output in references:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})

        additional_info = f"Info about the text: {context}\n" if context else ""
        messages.append(
            {"role": "user", "content": f"{additional_info}Text to classify: {sample}"}
        )

        result = completion(
            self.model,
            messages=messages,
            max_tokens=1,  #  NOTE: when using logit_bias for classification, max_tokens should be 1
            logit_bias=logit_bias_from_labels(
                self.model,
                set(self.mapping.keys()),
            ),
        )["choices"][0]["message"]["content"]

        return self.mapping.get(result, "UNKNOWN")

    def is_interactive(self) -> bool:
        return False
