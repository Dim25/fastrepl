import pytest

from typing import get_args
from evaluate import list_evaluation_modules

from fastrepl.eval.metric import load_metric
from fastrepl.eval.metric.huggingface import (
    HUGGINGFACE_BUILTIN_METRICS,
    HUGGINGFACE_FASTREPL_METRICS,
)

huggingface_builtin_metrics = list(get_args(HUGGINGFACE_BUILTIN_METRICS))
huggingface_fastrepl_metrics = list(get_args(HUGGINGFACE_FASTREPL_METRICS))


class TestHuggingfaceMetric:
    def test_all(self):
        assert huggingface_builtin_metrics == list_evaluation_modules(
            module_type="metric", include_community=False
        )

    @pytest.mark.parametrize(
        "name",
        huggingface_builtin_metrics + huggingface_fastrepl_metrics,
    )
    def test_metric(self, name):
        try:
            m = load_metric(name)
            if name in ["f1"]:
                # https://github.com/huggingface/evaluate/blob/af3c30561d840b83e54fc5f7150ea58046d6af69/metrics/f1/f1.py#L52
                result = m.compute(
                    predictions=[0, 0, 1, 1, 0], references=[0, 1, 0, 1, 0]
                )
                assert result == {"f1": 0.5}
            if name in ["exact_match"]:
                # https://github.com/huggingface/evaluate/blob/af3c30561d840b83e54fc5f7150ea58046d6af69/metrics/exact_match/exact_match.py#L50
                result = m.compute(
                    predictions=["the cat", "theater", "YELLING", "agent007"],
                    references=["cat?", "theater", "yelling", "agent"],
                )
                assert result == {"exact_match": 0.25}
        except NotImplementedError:
            pass
