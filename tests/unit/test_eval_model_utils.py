import pytest
import random
import warnings

import fastrepl.llm
from fastrepl.eval.model.utils import (
    logit_bias_from,
    mappings_from_labels,
    LabelMapping,
    next_mappings_for_consensus,
    warn_verbosity_bias,
)


@pytest.fixture(autouse=True)
def mock_tokenize(monkeypatch):
    def mock(model, text):
        if model == "command-nightly":
            return [ord(s) for s in text]
        elif model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]:
            return [ord(s) for s in text]
        raise NotImplementedError

    monkeypatch.setattr(fastrepl.llm, "tokenize", mock)


class TestLogitBiasForClassification:
    @pytest.mark.parametrize(
        "model, choices, expected",
        [
            (
                "gpt-3.5-turbo",
                "AB",
                {65: 100, 66: 100},
            ),
            (
                "gpt-4",
                "123",
                {49: 100, 50: 100, 51: 100},
            ),
        ],
    )
    def test_openai(self, model, choices, expected):
        actual = logit_bias_from(model, set(choices))
        assert actual == expected

    @pytest.mark.parametrize(
        "model, choices, expected",
        [
            (
                "command-nightly",
                "AB",
                {65: 10, 66: 10},
            ),
            (
                "command-nightly",
                "123",
                {49: 10, 50: 10, 51: 10},
            ),
        ],
    )
    def test_cohere(self, model, choices, expected):
        actual = logit_bias_from(model, set(choices))
        assert actual == expected

    @pytest.mark.parametrize("model", ["j2-ultra", "togethercomputer/llama-2-70b-chat"])
    def test_empty(self, model):
        assert logit_bias_from(model, "") == {}
        assert logit_bias_from(model, "ABC") == {}

    def test_not_single_token(self):
        with pytest.raises(ValueError):
            logit_bias_from("gpt-3.5-turbo", set(["GOOD", "GREAT"]))


def test_mapping_from_labels():
    mapping = mappings_from_labels(
        labels={
            "POSITIVE": "Given text is positive.",
            "NEGATIVE": "Given text is negative.",
            "NEUTRAL": "Given text is neutral.",
        },
        start=ord("A"),
        rg=random.Random(42),
    )
    assert mapping == [
        LabelMapping("A", "NEUTRAL", "Given text is neutral."),
        LabelMapping("B", "POSITIVE", "Given text is positive."),
        LabelMapping("C", "NEGATIVE", "Given text is negative."),
    ]


class TestNextMappingsForConsensus:
    @pytest.mark.parametrize(
        "mappings, result, expected",
        [
            (
                [
                    LabelMapping("A", "POSITIVE", "Given text is positive."),
                    LabelMapping("B", "NEGATIVE", "Given text is negative."),
                ],
                LabelMapping("B", "NEGATIVE", "Given text is negative."),
                None,
            ),
            (
                [
                    LabelMapping("A", "POSITIVE", "Given text is positive."),
                    LabelMapping("B", "NEGATIVE", "Given text is negative."),
                ],
                "B",
                None,
            ),
            (
                [
                    LabelMapping("A", "NEUTRAL", "Given text is neutral."),
                    LabelMapping("B", "POSITIVE", "Given text is positive."),
                    LabelMapping("C", "NEGATIVE", "Given text is negative."),
                ],
                "C",
                None,
            ),
        ],
    )
    def test_no_need(self, mappings, result, expected):
        assert next_mappings_for_consensus(mappings, result) == expected

    @pytest.mark.parametrize(
        "mappings, result, expected",
        [
            (
                [
                    LabelMapping("A", "POSITIVE", "Given text is positive."),
                    LabelMapping("B", "NEGATIVE", "Given text is negative."),
                ],
                LabelMapping("A", "POSITIVE", "Given text is positive."),
                [
                    LabelMapping("B", "NEGATIVE", "Given text is negative."),
                    LabelMapping("A", "POSITIVE", "Given text is positive."),
                ],
            ),
            (
                [
                    LabelMapping("A", "NEUTRAL", "Given text is neutral."),
                    LabelMapping("B", "POSITIVE", "Given text is positive."),
                    LabelMapping("C", "NEGATIVE", "Given text is negative."),
                ],
                "A",
                [
                    LabelMapping("C", "NEGATIVE", "Given text is negative."),
                    LabelMapping("B", "POSITIVE", "Given text is positive."),
                    LabelMapping("A", "NEUTRAL", "Given text is neutral."),
                ],
            ),
            (
                [
                    LabelMapping("A", "NEUTRAL", "Given text is neutral."),
                    LabelMapping("B", "POSITIVE", "Given text is positive."),
                    LabelMapping("C", "NEGATIVE", "Given text is negative."),
                    LabelMapping("D", "SOMETHING", "Given text is something."),
                ],
                "B",
                [
                    LabelMapping("D", "SOMETHING", "Given text is something."),
                    LabelMapping("C", "NEGATIVE", "Given text is negative."),
                    LabelMapping("A", "NEUTRAL", "Given text is neutral."),
                    LabelMapping("B", "POSITIVE", "Given text is positive."),
                ],
            ),
        ],
    )
    def test_need(self, mappings, result, expected):
        assert next_mappings_for_consensus(mappings, result) == expected


def test_warn_verbosity_bias():
    with pytest.warns() as record:
        warn_verbosity_bias(["A" * 9, "B" * 3, "C"])

    assert len(record) == 3
