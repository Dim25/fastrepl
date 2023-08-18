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


class TestSemanticAnswerSimilarityMetric:
    @pytest.mark.parametrize(
        "predictions, references, expected",
        [
            (
                [[""]],
                [[""]],
                {
                    "top_1_sas": [-6.185379],
                    "top_k_sas": [-6.185379],
                    "pred_label_matrix": [[[-6.1853790283203125]]],
                },
            ),
            (
                [
                    [
                        "This is an example sentence.",
                        "The quick brown fox jumps over the lazy dog.",
                        "I enjoy reading books in my free time.",
                    ]
                ],
                [
                    [
                        "An example sentence is provided.",
                        "A lazy dog is jumped over by a quick brown fox.",
                        "In my leisure time, I like to read books.",
                    ]
                ],
                {
                    "top_1_sas": [0.8043081, 0.9336777, 0.8648979],
                    "top_k_sas": [0.8043081, 0.9336777, 0.8648979],
                    "pred_label_matrix": [
                        [[0.8043081164360046]],
                        [[0.9336776733398438]],
                        [[0.8648979067802429]],
                    ],
                },
            ),
        ],
    )
    def test_cross_encoder(self, predictions, references, expected):
        m = load_metric(
            "sas", model_name_or_path="cross-encoder/ms-marco-TinyBERT-L-2-v2"
        )
        actual = m.compute(predictions=predictions, references=references)
        pytest.approx(expected, actual)

    @pytest.mark.parametrize(
        "predictions, references, expected",
        [
            (
                [[""]],
                [[""]],
                {
                    "top_1_sas": [-6.185379],
                    "top_k_sas": [-6.185379],
                    "pred_label_matrix": [[[-6.1853790283203125]]],
                },
            ),
            (
                [
                    [
                        "This is an example sentence.",
                        "The quick brown fox jumps over the lazy dog.",
                        "I enjoy reading books in my free time.",
                    ]
                ],
                [
                    [
                        "An example sentence is provided.",
                        "A lazy dog is jumped over by a quick brown fox.",
                        "In my leisure time, I like to read books.",
                    ]
                ],
                {
                    "top_1_sas": [9.87214, 9.589075, 2.323873],
                    "top_k_sas": [9.87214, 9.589075, 2.323873],
                    "pred_label_matrix": [
                        [[9.872139930725098]],
                        [[9.589075088500977]],
                        [[2.3238730430603027]],
                    ],
                },
            ),
        ],
    )
    def test_bi_encoder(self, predictions, references, expected):
        m = load_metric(
            "sas", model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"
        )
        actual = m.compute(predictions=predictions, references=references)
        pytest.approx(expected, actual)
