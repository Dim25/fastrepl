import random
import warnings
import functools
from typing import Optional, Tuple, Dict, List

from fastrepl.utils import prompt
from fastrepl.llm import completion, SUPPORTED_MODELS
from fastrepl.eval.base import BaseEvalWithoutReference

from fastrepl.eval.model.utils import (
    logit_bias_from_labels,
    mappings_from_labels,
    next_mappings_for_consensus,
    warn_verbosity_bias,
    LabelMapping,
    PositionDebiasStrategy,
)


@prompt
def system_prompt(context, labels, label_keys):
    """You are master of classification who can classify any text according to the user's instructions.
    {{context}}

    These are the labels you can use:
    {{labels}}

    Only output one of these label keys:
    {{label_keys}}"""


@prompt
def final_message_prompt(sample, context=""):
    """{% if context != '' %}
    Info about the text: {{ context }}
    {% endif %}
    Text to think about: {{ sample }}"""


class LLMClassifier(BaseEvalWithoutReference):
    def __init__(
        self,
        labels: Dict[str, str],
        context: str = "",
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
        rg=random.Random(42),
        references: List[Tuple[str, str]] = [],
        position_debias_strategy: PositionDebiasStrategy = "shuffle",
    ) -> None:
        warn_verbosity_bias(labels.values())

        self.labels = labels
        self.global_context = context
        self.model = model
        self.rg = rg
        self.references = references
        self.position_debias_strategy: PositionDebiasStrategy = position_debias_strategy

    def _compute(
        self,
        sample: str,
        context: str,
        mappings: List[LabelMapping],
        references: List[Tuple[str, str]],
    ) -> Optional[LabelMapping]:
        instruction = system_prompt(
            context=self.global_context,
            labels="\n".join(f"{m.token}: {m.description}" for m in mappings),
            label_keys=", ".join(m.token for m in mappings),
        )

        messages = [{"role": "system", "content": instruction}]
        for input, output in references:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})
        messages.append(
            {"role": "user", "content": final_message_prompt(sample, context)}
        )

        result = completion(
            self.model,
            messages=messages,
            max_tokens=1,  #  NOTE: when using logit_bias for classification, max_tokens should be 1
            logit_bias=logit_bias_from_labels(
                self.model, set(m.token for m in mappings)
            ),
        )["choices"][0]["message"]["content"]

        for m in mappings:
            if m.token == result:
                return m

        warnings.warn(f"classification result not in mapping: {result!r}")
        return None

    def compute(self, sample: str, context="") -> Optional[str]:
        references = self.rg.sample(self.references, len(self.references))
        mappings = mappings_from_labels(self.labels, rg=self.rg)

        result1 = self._compute(sample, context, mappings, references)
        if result1 is None:
            return None

        if self.position_debias_strategy == "shuffle":
            return result1.label

        next_mappings = next_mappings_for_consensus(mappings, result1)
        if next_mappings is None:
            return result1.label

        result2 = self._compute(sample, context, next_mappings, references)
        if result2 is None:
            return None

        return result1.label if result1.label == result2.label else None

    def is_interactive(self) -> bool:
        return False
