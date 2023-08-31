import pytest

from fastrepl import LLMClassifier


class TestEvalPositionConsensus:
    def test_shuffle(self, monkeypatch):
        pass

    def test_consensus(self, monkeypatch):
        class MockClassifier:
            def _compute(self, sample, context, mappings, references):
                return None

        monkeypatch.setattr(LLMClassifier, "_compute", MockClassifier._compute)

        eval = LLMClassifier(
            labels={
                "a": "this is a",
                "b": "this is b",
            },
            position_debias_strategy="consensus",
        )

        result = eval.compute("this is sample")
        assert result is None
