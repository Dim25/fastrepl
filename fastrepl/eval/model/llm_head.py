import random
import functools
import itertools

from abc import abstractmethod
from typing import Optional, Tuple, Iterable, TypedDict, List, Dict
from typing_extensions import Unpack, NotRequired

import fastrepl.llm as llm
from fastrepl.utils import prompt, number
from fastrepl.eval.base import BaseEvalNode

from fastrepl.warnings import (
    warn,
    VerbosityBiasWarning,
    InvalidPredictionWarning,
    FloatGradingWarning,
)
from fastrepl.eval.model.utils import (
    logit_bias_from,
    mappings_from_labels,
    next_mappings_for_consensus,
    check_length_inbalance,
    PositionDebiasStrategy,
)


class LLMEvaluationHeadParams(TypedDict):
    context: str
    options: NotRequired[Iterable[str]]  # TODO: It is actually required.
    model: NotRequired[llm.SUPPORTED_MODELS]
    rg: NotRequired[random.Random]
    references: NotRequired[List[Tuple[str, str]]]


class LLMEvaluationHead(BaseEvalNode):
    def __init__(self, **kwargs: Unpack[LLMEvaluationHeadParams]) -> None:
        self.global_context = kwargs["context"]
        self.options = kwargs["options"]
        self.model = kwargs.get("model", "gpt-3.5-turbo")
        self.rg = kwargs.get("rg", random.Random(42))
        self.references = kwargs.get("references", [])

    @abstractmethod
    def system_message(
        self, sample: str, global_context: str, local_context: Optional[str]
    ) -> Dict[str, str]:
        ...

    @abstractmethod
    def final_message(
        self, sample: str, global_context: str, local_context: Optional[str]
    ) -> Dict[str, str]:
        ...

    def reference_messages(
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

    def messages(
        self, sample: str, context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        system_message = self.system_message(sample, self.global_context, context)
        reference_messages = self.reference_messages(
            self.rg.sample(self.references, len(self.references))
        )
        final_message = self.final_message(sample, self.global_context, context)

        return [system_message, *reference_messages, final_message]

    def completion(self, sample: str, context: Optional[str] = None) -> Optional[str]:
        return llm.completion(
            model=self.model,
            messages=self.messages(sample, context),
            # NOTE: when using logit_bias for classification, max_tokens should be 1
            # max_tokens=2 is workaround. TODO
            max_tokens=2 if "together" in self.model else 1,
            logit_bias=logit_bias_from(self.model, [str(i) for i in self.options]),
        )["choices"][0]["message"]["content"]

    # NOTE: It is safe to return NONE, since metric will skip prediction-reference pair if prediction is NONE
    def compute(self, sample: str, context: Optional[str] = None) -> Optional[str]:
        result = self.completion(sample, context)
        if result is None:
            return None

        # we can get ' A' instead of 'A', which is still a single token.
        result = result.strip()

        # NOTE: Although we use max_token=1 and logit_bias, we still can get something different.
        # This is because 1. some LLM provider does not have logit_bias option
        # 2. for Cohere, max logit_bias value(=10) is not enough to force the model. (Not sure why.)
        if result not in self.options:
            warn(InvalidPredictionWarning, context=f"{result!r} not in {self.options}.")
            return None

        return result


class LLMClassificationHead(LLMEvaluationHead):
    def __init__(
        self,
        labels: Dict[str, str],
        position_debias_strategy: PositionDebiasStrategy = "shuffle",
        **kwargs: Unpack[LLMEvaluationHeadParams],
    ) -> None:
        if check_length_inbalance(labels.values()):  # pragma: no cover
            warn(VerbosityBiasWarning)

        self.labels = labels
        self.mapping = mappings_from_labels(labels)
        self.position_debias_strategy: PositionDebiasStrategy = position_debias_strategy

        kwargs.update({"options": [m.token for m in self.mapping]})
        super().__init__(**kwargs)

    def system_message(
        self, sample: str, global_context: str, local_context: Optional[str]
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

    def reference_messages(
        self, references: List[Tuple[str, str]]
    ) -> List[Dict[str, str]]:
        def label2token(text: str) -> str:
            return functools.reduce(
                lambda t, m: t.replace(m.label, m.token), self.mapping, text
            )

        return super().reference_messages(
            [(input, label2token(output)) for input, output in references]
        )

    def final_message(
        self, sample: str, global_context: str, local_context: Optional[str]
    ) -> Dict[str, str]:
        @prompt
        def p(sample, context):
            """{% if context is not none %}
            Info about the text: {{ context }}
            {% endif %}
            Text to classify: {{ sample }}"""

        return {
            "role": "user",
            "content": p(sample, local_context),
        }

    def _compute(self, sample: str, context: Optional[str]) -> Optional[str]:
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
        self.number_from, self.number_to = number_from, number_to
        kwargs.update({"options": [str(i) for i in range(number_from, number_to + 1)]})
        super().__init__(**kwargs)

    def system_message(
        self, sample: str, global_context: str, local_context: Optional[str]
    ) -> Dict[str, str]:
        @prompt
        def p(context):
            """You are master of grading who can grade any text according to the user's instructions. Only output a single integer.
            {{context}}"""

        return {"role": "system", "content": p(global_context)}

    def final_message(
        self, sample: str, global_context: str, local_context: Optional[str]
    ) -> Dict[str, str]:
        @prompt
        def p(sample, context):
            """{% if context is not none %}
            Info about the text: {{ context }}
            {% endif %}
            Text to grade: {{ sample }}"""

        return {"role": "user", "content": p(sample, local_context)}

    def compute(self, sample: str, context: Optional[str] = None) -> Optional[str]:
        result = number(self.completion(sample, context))
        if result is None:
            return None

        if result < self.number_from or result > self.number_to:
            warn(
                InvalidPredictionWarning,
                context=f"{result!r} is not in range [{self.number_from}, {self.number_to}].",
            )
            return None

        if type(result) is float:
            warn(FloatGradingWarning, context=f"{result!r} is not an integer.")

        return str(result)
