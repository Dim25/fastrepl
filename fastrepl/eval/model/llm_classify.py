import random
from typing import Tuple, Dict, List

from fastrepl.run import completion, SUPPORTED_MODELS
from fastrepl.eval.model.base import BaseModelEval
from fastrepl.eval.model.utils import render_labels, logit_bias_from_labels


class LLMClassifier(BaseModelEval):
    __slots__ = ("labels", "model", "rg", "references", "system_msg")

    def __init__(
        self,
        labels: Dict[str, str],
        context: str = "",
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
        rg=random.Random(42),
        references: List[Tuple[str, str]] = [],
    ) -> None:
        self.labels = labels
        self.model = model
        self.rg = rg
        self.references = references

        self.system_msg = {
            "role": "system",
            "content": f"""You are master of classification who can classify any text according to the user's instructions.
{context}

These are the labels you can use:
{render_labels(self.labels)}

Only output one of these label keys:
{self.labels.keys()}""",
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

        result = (
            completion(
                self.model,
                messages=messages,
                max_tokens=1,  #  NOTE: when using logit_bias for classification, max_tokens should be 1
                logit_bias=logit_bias_from_labels(
                    self.model,
                    set(self.labels.keys()),
                ),
            )
            .choices[0]
            .message.content
        )
        return self.labels.get(result, "UNKNOWN")
