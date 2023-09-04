import pytest
from typing import Optional

from fastrepl import Evaluator
from fastrepl.errors import EmptyPipelineError
from fastrepl.eval.base import BaseEvalNode


class MockEval(BaseEvalNode):
    def compute(self, sample: str, context: Optional[str] = None) -> str:
        return ("" if context is None else context) + sample + "0"


class TestEvaluator:
    def test_empty_pipeline(self):
        with pytest.raises(EmptyPipelineError):
            Evaluator(pipeline=[])

    @pytest.mark.parametrize(
        "pipeline, sample, context, result",
        [
            ([MockEval()], "1", "2", "210"),
            ([MockEval(), MockEval()], "1", "2", "21010"),
            ([MockEval(), MockEval(), MockEval()], "1", "2", "2101010"),
        ],
    )
    def test_with_context(self, pipeline, sample, context, result):
        assert Evaluator(pipeline).run(sample=sample, context=context) == result

    @pytest.mark.parametrize(
        "pipeline, sample, result",
        [
            ([MockEval()], "1", "10"),
            ([MockEval(), MockEval()], "1", "1010"),
            ([MockEval(), MockEval(), MockEval()], "1", "101010"),
        ],
    )
    def test_without_context(self, pipeline, sample, result):
        assert Evaluator(pipeline).run(sample=sample) == result
