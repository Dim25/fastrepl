import pytest

import fastrepl
from fastrepl.eval.model.base import BaseModelEval


class TestEvaluator:
    def test_empty_pipeline(self):
        with pytest.raises(ValueError):
            fastrepl.Evaluator(pipeline=[])

    def test_single_node(self):
        class MockEval(BaseModelEval):
            def compute(self, sample: str, context="") -> str:
                return context + sample + "0"

        pipeline = [MockEval()]
        assert fastrepl.Evaluator(pipeline).run("1", "2") == "210"

    def test_two_node(self):
        class MockEval(BaseModelEval):
            def compute(self, sample: str, context="") -> str:
                return context + sample + "0"

        pipeline = [MockEval(), MockEval()]
        assert fastrepl.Evaluator(pipeline).run("1", "2") == "21010"
