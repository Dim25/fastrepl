import pytest

from fastrepl.run.llm import tokenize
from fastrepl.run import logit_bias_for_classification


class TestTokenize:
    @pytest.mark.parametrize(
        "model, text, expected",
        [
            ("gpt-3.5-turbo", "A", [32]),
            ("gpt-3.5-turbo", "B", [33]),
            ("gpt-3.5-turbo", "C", [34]),
            ("gpt-3.5-turbo", "1", [16]),
            ("gpt-3.5-turbo", "2", [17]),
            ("gpt-3.5-turbo", "3", [18]),
            #
            ("gpt-3.5-turbo-16k", "A", [32]),
            ("gpt-3.5-turbo-16k", "B", [33]),
            ("gpt-3.5-turbo-16k", "C", [34]),
            ("gpt-3.5-turbo-16k", "1", [16]),
            ("gpt-3.5-turbo-16k", "2", [17]),
            ("gpt-3.5-turbo-16k", "3", [18]),
            #
            ("gpt-4", "A", [32]),
            ("gpt-4", "B", [33]),
            ("gpt-4", "C", [34]),
            ("gpt-4", "1", [16]),
            ("gpt-4", "2", [17]),
            ("gpt-4", "3", [18]),
        ],
    )
    def test_openai(self, model, text, expected):
        actual = tokenize(model, text)
        assert actual == expected

    @pytest.mark.parametrize(
        "model, text, expected",
        [
            ("command-nightly", "A", [40]),
            ("command-nightly", "B", [41]),
            ("command-nightly", "C", [42]),
            ("command-nightly", "1", [24]),
            ("command-nightly", "2", [25]),
            ("command-nightly", "3", [26]),
        ],
    )
    def test_cohere(self, model, text, expected):
        actual = tokenize(model, text)
        assert actual == expected

    def test_ai21(self):
        with pytest.raises(NotImplementedError):
            tokenize("j2-ultra", "A")


class TestLogitBias:
    @pytest.mark.parametrize(
        "model, choices, expected",
        [
            (
                "gpt-3.5-turbo",
                "ABCDE",
                {32: 100, 33: 100, 34: 100, 35: 100, 36: 100},
            ),
            (
                "gpt-3.5-turbo",
                "123",
                {16: 100, 17: 100, 18: 100},
            ),
            (
                "gpt-3.5-turbo-16k",
                "ABCDE",
                {32: 100, 33: 100, 34: 100, 35: 100, 36: 100},
            ),
            (
                "gpt-3.5-turbo-16k",
                "123",
                {16: 100, 17: 100, 18: 100},
            ),
            (
                "gpt-4",
                "ABCDE",
                {32: 100, 33: 100, 34: 100, 35: 100, 36: 100},
            ),
            (
                "gpt-4",
                "123",
                {16: 100, 17: 100, 18: 100},
            ),
        ],
    )
    def test_openai(self, model, choices, expected):
        actual = logit_bias_for_classification(model, choices)
        assert actual == expected

    @pytest.mark.parametrize(
        "model, choices, expected",
        [
            (
                "command-nightly",
                "ABCDE",
                {40: 10, 41: 10, 42: 10, 43: 10, 44: 10},
            ),
            (
                "command-nightly",
                "123",
                {24: 10, 25: 10, 26: 10},
            ),
        ],
    )
    def test_cohere(self, model, choices, expected):
        actual = logit_bias_for_classification(model, choices)
        assert actual == expected

    def test_ai21(self):
        with pytest.raises(NotImplementedError):
            logit_bias_for_classification("j2-ultra", "ABC")

    def test_invalid(self):
        with pytest.raises(ValueError):
            logit_bias_for_classification("gpt-3.5-turbo", "ABCA")
