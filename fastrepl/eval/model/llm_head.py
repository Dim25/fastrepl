import random
import itertools
from abc import abstractmethod
from typing import Optional, Tuple, Iterable, TypedDict, List, Dict
from typing_extensions import Unpack, NotRequired

from fastrepl.utils import prompt
from fastrepl.llm import completion, SUPPORTED_MODELS
from fastrepl.eval.base import BaseEvalWithoutReference

from fastrepl.eval.model.utils import (
    logit_bias_from,
    mappings_from_labels,
    next_mappings_for_consensus,
    warn_verbosity_bias,
    PositionDebiasStrategy,
)


class LLMEvaluationHeadParams(TypedDict):
    context: str
    options: Iterable[str]
    model: NotRequired[SUPPORTED_MODELS]
    rg: NotRequired[random.Random]
    references: NotRequired[List[Tuple[str, str]]]


class LLMEvaluationHead(BaseEvalWithoutReference):
    def __init__(self, **kwargs: Unpack[LLMEvaluationHeadParams]) -> None:
        self.global_context = kwargs["context"]
        self.options = kwargs["options"]
        self.model = kwargs.get("model", "gpt-3.5-turbo")
        self.rg = kwargs.get("rg", random.Random(42))
        self.references = kwargs.get("references", [])

    @abstractmethod
    def _system_message(
        self, sample: str, global_context: str, local_context: str
    ) -> Dict[str, str]:
        ...

    @abstractmethod
    def _final_message(
        self, sample: str, global_context: str, local_context: str
    ) -> Dict[str, str]:
        ...

    def _reference_messages(
        self,
        references: List[Tuple[str, str]],
    ) -> List[Dict[str, str]]:
        return list(
            itertools.chain.from_iterable(
                (
                    {"role": "user", "content": input},
                    {"role": "assistant", "content": output},
                )
                for input, output in references
            )
        )

    def compute(self, sample: str, context: Optional[str] = None) -> Optional[str]:
        messages = [
            self._system_message(sample, self.global_context, context),
            *self._reference_messages(
                self.rg.sample(self.references, len(self.references))
            ),
            self._final_message(sample, self.global_context, context),
        ]

        result = completion(
            model=self.model,
            messages=messages,
            max_tokens=1,  # NOTE: when using logit_bias for classification, max_tokens should be 1
            logit_bias=logit_bias_from(self.model, [str(i) for i in self.options]),
        )["choices"][0]["message"]["content"]

        # NOTE: Some LLM provider does not have logit_bias option
        return result if result in self.options else None


class LLMClassificationHead(LLMEvaluationHead):
    def __init__(
        self,
        labels: Dict[str, str],
        position_debias_strategy: PositionDebiasStrategy = "shuffle",
        **kwargs: Unpack[LLMEvaluationHeadParams],
    ) -> None:
        warn_verbosity_bias(labels.values())

        self.labels = labels
        self.mapping = mappings_from_labels(labels)
        self.position_debias_strategy: PositionDebiasStrategy = position_debias_strategy

        kwargs.update({"options": [m.token for m in self.mapping]})
        super().__init__(**kwargs)

    def _system_message(
        self, sample: str, global_context: str, local_context: str
    ) -> Dict[str, str]:
        @prompt
        def p(context, labels, label_keys):
            """You are master of classification who can classify any text according to the user's instructions.
            {{context}}

            These are the labels you can use:
            {{labels}}

            Only output one of these label keys:
            {{label_keys}}"""

        return {
            "role": "system",
            "content": p(
                context=global_context,
                labels="\n".join(f"{m.token}: {m.description}" for m in self.mapping),
                label_keys=", ".join(m.token for m in self.mapping),
            ),
        }

    def _reference_messages(
        self, references: List[Tuple[str, str]]
    ) -> List[Dict[str, str]]:
        def label2token(label: str) -> str:
            return next(m.token for m in self.mapping if m.label == label)

        return super()._reference_messages(
            [(input, label2token(output)) for input, output in references]
        )

    def _final_message(
        self, sample: str, global_context: str, local_context: str
    ) -> Dict[str, str]:
        @prompt
        def p(sample, context):
            """{% if context is not None %}
            Info about the text: {{ context }}
            {% endif %}
            Text to classify: {{ sample }}"""

        return {
            "role": "user",
            "content": p(sample, local_context),
        }

    def _compute(self, sample: str, context="") -> Optional[str]:
        if self.position_debias_strategy == "shuffle":
            self.mapping = mappings_from_labels(self.labels, rg=self.rg)
            return super().compute(sample, context)

        if self.position_debias_strategy == "consensus":
            initial_result = super().compute(sample, context)
            if initial_result is None:
                return None

            next_mapping = next_mappings_for_consensus(self.mapping, initial_result)
            if next_mapping is None:
                return initial_result

            self.mapping = next_mapping
            next_result = super().compute(sample, context)
            if next_result is None:
                return None

            return initial_result if initial_result == next_result else None

    def compute(self, sample: str, context: Optional[str] = None) -> Optional[str]:
        token = self._compute(sample, context)
        if token is None:
            return None

        return next(m.label for m in self.mapping if m.token == token)


class LLMGradingHead(LLMEvaluationHead):
    def __init__(
        self,
        number_from: int,
        number_to: int,
        **kwargs: Unpack[LLMEvaluationHeadParams],
    ) -> None:
        kwargs.update({"options": [str(i) for i in range(number_from, number_to + 1)]})
        super().__init__(**kwargs)

    def _system_message(
        self, sample: str, global_context: str, local_context: str
    ) -> Dict[str, str]:
        @prompt
        def p(context):
            """You are master of grading who can grade any text according to the user's instructions.
            {{context}}"""

        return {"role": "system", "content": p(global_context)}

    def _final_message(
        self, sample: str, global_context: str, local_context: str
    ) -> Dict[str, str]:
        @prompt
        def p(sample, context):
            """{% if context is not None %}
            Info about the text: {{ context }}
            {% endif %}
            Text to grade: {{ sample }}"""

        return {"role": "user", "content": p(sample, local_context)}
