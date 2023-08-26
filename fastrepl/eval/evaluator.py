import functools
from typing import List

from fastrepl.eval.model.base import BaseModelEval


# TODO: we assume only model-eval here.
class Evaluator:
    __slots__ = ["pipeline"]

    def __init__(self, pipeline: List[BaseModelEval]) -> None:
        if len(pipeline) == 0:
            raise ValueError("Pipeline cannot be empty")
        self.pipeline = pipeline

    def run(self, sample: str, initial_context="") -> str:
        return functools.reduce(
            lambda context, eval: eval.compute(sample, context),
            self.pipeline,
            initial_context,
        )
