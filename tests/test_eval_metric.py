import pytest

from typing import get_args
from evaluate import list_evaluation_modules
import numpy as np

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
                    "top_1_sas": [8.154116],
                    "top_k_sas": [8.154116],
                    "pred_label_matrix": [[[8.154115676879883]]],
                },
            ),
            (
                [
                    ["She adores reading classic novels."],
                    ["The painting features vivid colors."],
                    ["The sun rises in the east."],
                    ["Good health is above wealth."],
                ],
                [
                    ["She loves perusing timeless literature."],
                    ["The artwork showcases vibrant hues."],
                    ["Bananas are a great source of potassium."],
                    ["Can I have a cup of coffee?"],
                ],
                {
                    "top_1_sas": [2.8915386, 3.4521167, -11.099813, -11.315325],
                    "top_k_sas": [2.8915386, 3.4521167, -11.099813, -11.315325],
                    "pred_label_matrix": [
                        [[2.891538619995117]],
                        [[3.4521167278289795]],
                        [[-11.099813461303711]],
                        [[-11.315324783325195]],
                    ],
                },
            ),
        ],
    )
    def test_cross_encoder(self, predictions, references, expected):
        m = load_metric(
            "sas", model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2"
        )
        actual = m.compute(predictions=predictions, references=references)
        assert np.allclose(actual["top_1_sas"], expected["top_1_sas"])
        assert np.allclose(actual["top_k_sas"], expected["top_k_sas"])
        assert np.allclose(actual["pred_label_matrix"], expected["pred_label_matrix"])

    @pytest.mark.parametrize(
        "predictions, references, expected",
        [
            (
                [[""]],
                [[""]],
                {
                    "top_1_sas": [0.99999994],
                    "top_k_sas": [0.99999994],
                    "pred_label_matrix": [[[0.9999999403953552]]],
                },
            ),
            (
                [
                    ["She adores reading classic novels."],
                    ["The painting features vivid colors."],
                    ["The sun rises in the east."],
                    ["Good health is above wealth."],
                ],
                [
                    ["She loves perusing timeless literature."],
                    ["The artwork showcases vibrant hues."],
                    ["Bananas are a great source of potassium."],
                    ["Can I have a cup of coffee?"],
                ],
                {
                    "top_1_sas": [0.77481174, 0.7642534, 0.028543131, 0.12256175],
                    "top_k_sas": [0.77481174, 0.7642534, 0.028543131, 0.12256175],
                    "pred_label_matrix": [
                        [[0.7748117446899414]],
                        [[0.7642533779144287]],
                        [[0.028543131425976753]],
                        [[0.1225617527961731]],
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
        assert np.allclose(actual["top_1_sas"], expected["top_1_sas"])
        assert np.allclose(actual["top_k_sas"], expected["top_k_sas"])
        assert np.allclose(actual["pred_label_matrix"], expected["pred_label_matrix"])
