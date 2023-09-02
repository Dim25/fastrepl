import pytest
from datasets import Dataset

import fastrepl
from fastrepl.utils import getenv


def _mapper(label):
    if label == "POSITIVE":
        return 1
    elif label == "NEGATIVE":
        return 0
    else:
        raise ValueError("Invalid label")


@pytest.mark.fastrepl
def test_api_base():
    import litellm

    assert getenv("LITELLM_PROXY_API_BASE", "") != ""
    assert litellm.api_base is not None


class TestEvaluationHead:
    @pytest.mark.fastrepl
    def test_single_classifier(self):
        labels = {
            "POSITIVE": "Given text is actually negative",
            "NEGATIVE": "Given text is actually positive",
        }

        eval = fastrepl.Evaluator(
            pipeline=[
                fastrepl.LLMClassificationHead(
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
