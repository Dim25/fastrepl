from fastrepl.eval.model import (
    LLMClassifier,
    LLMChainOfThought,
    LLMChainOfThoughtClassifier,
)


class TestClassifier:
    def test_single_classifier(self):
        labels = {
            "POSITIVE": "Given text is actually positive",
            "NEGATIVE": "Given text is actually negative",
        }
        eval = LLMClassifier(
            model="gpt-3.5-turbo",
            labels=labels,
        )

        assert eval.compute("What a great day!") in labels.keys()
        assert eval.compute("What a bad day!") in labels.keys()

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

        thought = pipeline[0].compute(input)
        answer = pipeline[1].compute(input, context=thought)
        assert answer in labels.keys()

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
        assert eval.compute("What a great day! I am so happy.") in labels.keys()
        assert eval.compute("What a bad day! I am so sad.") in labels.keys()
