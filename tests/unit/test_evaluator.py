import pytest
from datasets import Dataset

from fastrepl.loop import Evaluator
from fastrepl.eval.model.base import BaseModelEval


class TestEvaluator:
    def test_no_input(self):
        ds = Dataset.from_dict({"a": [1]})

        with pytest.raises(ValueError):
            Evaluator(
                dataset=ds,
                evals=[],
            )

    def test_features(self):
        input_ds = Dataset.from_dict({"input": [1]})
        output_ds = Evaluator(
            dataset=input_ds,
            evals=[],
        ).run()

        assert len(output_ds.features) == 2
        assert "input" in output_ds.features
        assert "output" in output_ds.features

    def test_eval_pipe_single(self):
        class MockEval(BaseModelEval):
            def compute(self, sample: str, context="") -> str:
                return context + sample + "0"

        input_ds = Dataset.from_dict({"input": ["1"]})
        output_ds = Evaluator(
            dataset=input_ds,
            evals=[MockEval(), MockEval()],
        ).run()

        assert output_ds["output"] == ["1010"]

    def test_eval_pipe_multiple(self):
        class MockEval(BaseModelEval):
            def compute(self, sample: str, context="") -> str:
                return context + sample + "0"

        input_ds = Dataset.from_dict({"input": ["1", "2", "3"]})
        output_ds = Evaluator(
            dataset=input_ds,
            evals=[MockEval(), MockEval()],
        ).run()

        assert output_ds["output"] == ["1010", "2020", "3030"]
