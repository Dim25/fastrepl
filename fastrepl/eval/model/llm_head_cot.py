from typing import Optional, Dict

from fastrepl.utils import prompt
from fastrepl.llm import completion
from fastrepl.eval.model import LLMClassificationHead, LLMGradingHead
from fastrepl.warnings import warn, InvalidPredictionWarning


class LLMClassificationHeadCOT(LLMClassificationHead):
    def system_message(
        self, sample: str, global_context: str, local_context: Optional[str]
    ) -> Dict[str, str]:
        @prompt
        def p(context, labels):
            """You are master of classification who can classify any text according to the user's instructions.
            When user give you the text to classify, you do step-by-step thinking within 3 sentences and give a final result.

            When doing step-by-step thinking, you must consider the following:
            {{ context }}

            These are the labels(KEY: DESCRIPTION) you can use:
            {{labels}}

            Your response must strictly follow this format:
            ### Thoughts
            <STEP_BY_STEP_THOUGHTS>
            ### Result
            <SINGLE_LABEL_KEY>"""

        return {
            "role": "system",
            "content": p(
                context=global_context,
                labels="\n".join(f"{m.token}: {m.description}" for m in self.mapping),
            ),
        }

    def final_message(
        self, sample: str, global_context: str, local_context: Optional[str]
    ) -> Dict[str, str]:
        @prompt
        def p(sample, context):
            """{% if context is not none %}
            Info about the text: {{ context }}
            {% endif %}
            Text to think and clasify: {{ sample }}"""

        return {"role": "user", "content": p(sample, local_context)}

    def compute(self, sample: str, context: Optional[str] = None) -> Optional[str]:
        # fmt: off
        result = completion(
            model=self.model,
            messages=self.messages(sample, context)
        )["choices"][0]["message"]["content"]
        # fmt: on

        prediction = result.split("### Result")[-1].strip()

        if prediction not in self.options:
            warn(
                InvalidPredictionWarning,
                context=f"{prediction!r} not in {self.options}. full: {result}",
            )
            return None

        return next(m.label for m in self.mapping if m.token == prediction)


class LLMGradingHeadCOT(LLMGradingHead):
    def system_message(
        self, sample: str, global_context: str, local_context: Optional[str]
    ) -> Dict[str, str]:
        @prompt
        def p(context):
            """You are master of grading who can grade any text according to the user's instructions.
            When user give you the text to grade, you do step-by-step thinking within 5 sentences and give a final result.

            When doing step-by-step thinking, you must consider the following:
            {{ context }}

            Your response must strictly follow this format:
            ### Thoughts
            <STEP_BY_STEP_THOUGHTS>
            ### Result
            <NUMBER>"""

        return {"role": "system", "content": p(global_context)}

    def final_message(
        self, sample: str, global_context: str, local_context: Optional[str]
    ) -> Dict[str, str]:
        @prompt
        def p(sample, context):
            """{% if context is not none %}
            Info about the text: {{ context }}
            {% endif %}
            Text to think and grade: {{ sample }}"""

        return {"role": "user", "content": p(sample, local_context)}

    def compute(self, sample: str, context: Optional[str] = None) -> Optional[str]:
        # fmt: off
        result = completion(
            model=self.model,
            messages=self.messages(sample, context)
        )["choices"][0]["message"]["content"]
        # fmt: on

        prediction = result.split("### Result")[-1].strip()

        if prediction not in self.options:
            warn(
                InvalidPredictionWarning,
                context=f"{prediction!r} not in {self.options}. full: {result}",
            )
            return None

        return prediction
