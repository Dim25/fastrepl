import pytest

import fastrepl
from fastrepl.eval.base import BaseEvalWithoutReference


class MockEval(BaseEvalWithoutReference):
    def compute(self, sample: str, context="") -> str:
        return context + sample + "0"


class TestEvaluator:
    def test_empty_pipeline(self):
        with pytest.raises(ValueError):
            fastrepl.Evaluator(pipeline=[])

    def test_single_node(self):
        pipeline = [MockEval()]
        assert fastrepl.Evaluator(pipeline).run("1", "2") == "210"

    def test_two_node(self):
        pipeline = [MockEval(), MockEval()]
        assert fastrepl.Evaluator(pipeline).run("1", "2") == "21010"
