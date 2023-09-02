import functools
from typing import Optional, List

from fastrepl.eval.base import BaseEvalWithoutReference


class Evaluator:
    def __init__(self, pipeline: List[BaseEvalWithoutReference]) -> None:
        if len(pipeline) == 0:
            raise ValueError("Pipeline cannot be empty")
        self.pipeline = pipeline

    def run(self, sample: str, initial_context="") -> Optional[str]:
        return functools.reduce(
            lambda context, eval: eval.compute(sample, context),
            self.pipeline,
            initial_context,
        )
