import pytest
from datasets import Dataset

import fastrepl


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
            "POSITIVE": "Given text is actually negative",
            "NEGATIVE": "Given text is actually positive",
        }

        eval = fastrepl.Evaluator(
            pipeline=[
                fastrepl.LLMClassifier(
                    model="gpt-3.5-turbo",
                    context="You will get a input text by a liar. Take it as the opposite.",
                    labels=labels,
                )
            ]
        )

        ds = Dataset.from_dict(
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

        result = fastrepl.LocalRunner(evaluator=eval, dataset=ds).run()

        predictions = [_mapper(label) for label in result["prediction"]]
        references = [_mapper(label) for label in result["reference"]]

        metric = fastrepl.load_metric("accuracy")
        assert metric.compute(predictions, references)["accuracy"] > -1

    @pytest.mark.fastrepl
    def test_cot_with_classifier(self):
        labels = {
            "POSITIVE": "Given text is actually negative",
            "NEGATIVE": "Given text is actually positive",
        }

        eval = fastrepl.Evaluator(
            pipeline=[
                fastrepl.LLMChainOfThought(
                    model="gpt-3.5-turbo",
                    labels=labels,
                    context="You will get a input text by a liar. Take it as the opposite.",
                ),
                fastrepl.LLMClassifier(
                    model="gpt-3.5-turbo",
                    context="You will get a input text by a liar. Take it as the opposite.",
                    labels=labels,
                ),
            ]
        )

        ds = Dataset.from_dict(
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

        result = fastrepl.LocalRunner(evaluator=eval, dataset=ds).run()

        predictions = [_mapper(label) for label in result["prediction"]]
        references = [_mapper(label) for label in result["reference"]]

        metric = fastrepl.load_metric("accuracy")
        assert metric.compute(predictions, references)["accuracy"] > -1

    @pytest.mark.fastrepl
    def test_cot_and_classify(self):
        eval = fastrepl.Evaluator(
            pipeline=[
                fastrepl.LLMChainOfThoughtClassifier(
                    model="gpt-3.5-turbo",
                    context="You will get a input text by a liar. Take it as the opposite.",
                    labels={
                        "POSITIVE": "Given text is actually negative",
                        "NEGATIVE": "Given text is actually positive",
                    },
                )
            ]
        )

        ds = Dataset.from_dict(
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

        result = fastrepl.LocalRunner(evaluator=eval, dataset=ds).run()

        predictions = [_mapper(label) for label in result["prediction"]]
        references = [_mapper(label) for label in result["reference"]]

        metric = fastrepl.load_metric("accuracy")
        assert metric.compute(predictions, references)["accuracy"] > -1
