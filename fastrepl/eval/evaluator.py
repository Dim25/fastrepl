import functools
from typing import Optional, List

from fastrepl.errors import EmptyPipelineError
from fastrepl.eval.base import BaseEvalNode


class Evaluator:
    def __init__(self, pipeline: List[BaseEvalNode]) -> None:
        if len(pipeline) == 0:
            raise EmptyPipelineError
        self.pipeline = pipeline

    def run(self, sample: str, context: Optional[str] = None) -> Optional[str]:
        initial_context = context

        return functools.reduce(
            lambda context, eval: eval.compute(sample, context),
            self.pipeline,
            initial_context,
        )
