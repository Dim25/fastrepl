import random
from typing import Tuple, Dict, List

from fastrepl.llm import completion, SUPPORTED_MODELS
from fastrepl.eval.model.base import BaseModelEval
from fastrepl.eval.model.utils import render_labels, mapping_from_labels

LLM_COT_CLASSIFY_SYSTEM_TPL = """You are master of classification who can classify any text according to the user's instructions.
If user gave you the text, do step by step thinking first, and classify it.

When do step-by-step thinking(less than 30 words), you must consider the following:
{context}

These are the labels you can use:
{labels}

For classification, only output one of these label keys:
{label_keys}

When responding, strictly follow this format: ### Thoghts: <STEP_BY_STEP_THOUGHTS>\n### Label: <LABEL>"""


class LLMChainOfThoughtClassifier(BaseModelEval):
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
            "content": LLM_COT_CLASSIFY_SYSTEM_TPL.format(
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
        # fmt: off
        result = completion(
            model=self.model,
            messages=messages, 
        )["choices"][0]["message"]["content"]
        # fmt: on

        return self.mapping.get(result[-1], "UNKNOWN")
