import pytest
from datasets import Dataset

from fastrepl.eval.metric import load_metric
from fastrepl.eval.model import (
    LLMClassifier,
    LLMChainOfThought,
    LLMChainOfThoughtClassifier,
)


def _mapper(label):
    if label == "POSITIVE":
        return 1
    elif label == "NEGATIVE":
        return 0
    else:
        raise ValueError("Invalid label")


class TestClassifier:
    @pytest.mark.fastrepl
    def test_single_classifier(self):
        labels = {
            "POSITIVE": "Given text is actually positive",
            "NEGATIVE": "Given text is actually negative",
        }
        eval = LLMClassifier(
            model="gpt-3.5-turbo",
            context="You will get a input text by a liar. Take it as the opposite.",
            labels=labels,
        )

        tc = Dataset.from_dict(
            {
                "input": [
                    "What a great day!",
                    "What a bad day!",
                    "I am so happy.",
                    "I am so sad.",
                ],
                "reference": [
                    "NEGATIVE",
                    "POSITIVE",
                    "NEGATIVE",
                    "POSITIVE",
                ],
            }
        )

        predictions = [eval.compute(input) for input in tc["input"]]
        references = tc["reference"]

        predictions = [_mapper(label) for label in predictions]
        references = [_mapper(label) for label in references]

        assert (
            load_metric("accuracy").compute(predictions, references)["accuracy"] > 0.5
        )

    @pytest.mark.fastrepl
    def test_cot_with_classifier(self):
        input = "What a great day!"
        labels = {
            "POSITIVE": "Given text is actually positive",
            "NEGATIVE": "Given text is actually negative",
        }

        pipeline = [
            LLMChainOfThought(
                model="gpt-3.5-turbo",
                labels=labels,
                context="You will get a input text by a liar. Take it as the opposite.",
            ),
            LLMClassifier(
                model="gpt-4",
                labels=labels,
            ),
        ]

        tc = Dataset.from_dict(
            {
                "input": [
                    "What a great day!",
                    "What a bad day!",
                    "I am so happy.",
                    "I am so sad.",
                ],
                "reference": [
                    "NEGATIVE",
                    "POSITIVE",
                    "NEGATIVE",
                    "POSITIVE",
                ],
            }
        )

        def predict(input):
            thought = pipeline[0].compute(input)
            return pipeline[1].compute(input, context=thought)

        predictions = [predict(input) for input in tc["input"]]
        references = tc["reference"]

        predictions = [_mapper(label) for label in predictions]
        references = [_mapper(label) for label in references]

        assert (
            load_metric("accuracy").compute(predictions, references)["accuracy"] > 0.5
        )

    @pytest.mark.fastrepl
    def test_cot_and_classify(self):
        labels = {
            "POSITIVE": "Given text is actually positive",
            "NEGATIVE": "Given text is actually negative",
        }
        eval = LLMChainOfThoughtClassifier(
            model="gpt-3.5-turbo",
            context="You will get a input text by a liar. Take it as the opposite.",
            labels=labels,
        )

        tc = Dataset.from_dict(
            {
                "input": [
                    "What a great day!",
                    "What a bad day!",
                    "I am so happy.",
                    "I am so sad.",
                ],
                "reference": [
                    "NEGATIVE",
                    "POSITIVE",
                    "NEGATIVE",
                    "POSITIVE",
                ],
            }
        )

        predictions = [eval.compute(input) for input in tc["input"]]
        references = tc["reference"]

        predictions = [_mapper(label) for label in predictions]
        references = [_mapper(label) for label in references]

        assert (
            load_metric("accuracy").compute(predictions, references)["accuracy"] > 0.5
        )
