import random
from typing import Tuple, List, Dict

from fastrepl.run import completion, SUPPORTED_MODELS
from fastrepl.eval.model.base import BaseModelEval
from fastrepl.eval.model.utils import render_labels


class LLMChainOfThought(BaseModelEval):
    __slots__ = ("model", "references", "rg", "system_msg")

    def __init__(
        self,
        context: str,
        labels: Dict[str, str],
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
        rg=random.Random(42),
        references: List[Tuple[str, str]] = [],
    ) -> None:
        self.model = model
        self.references = references
        self.rg = rg
        self.system_msg = {
            "role": "system",
            "content": f"""If user gave you the text, do step by step thinking that is needed to classify it.
Use less than 50 words.

These are the labels that will be used later to classify the text:
{render_labels(labels)}

When do step-by-step thinking, you must consider the following:
{context}

### Thoghts:""",
        }

    def compute(self, sample: str, context="") -> str:
        references = self.rg.sample(self.references, len(self.references))

        messages = [self.system_msg]
        for input, output in references:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})

        additional_info = f"Info about the text: {context}\n" if context else ""
        messages.append(
            {
                "role": "user",
                "content": f"{additional_info}Text to think about: {sample}",
            }
        )

        result = completion(self.model, messages=messages).choices[0].message.content
        return result