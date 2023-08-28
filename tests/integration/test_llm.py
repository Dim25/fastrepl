import pytest

import fastrepl
from fastrepl.llm import completion, tokenize


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


def test_gpt_cache():
    fastrepl.LLMCache.enable()
    assert fastrepl.LLMCache.enabled() == True

    result1 = completion("gpt-3.5-turbo", [{"role": "user", "content": "hi"}])
    result2 = completion("gpt-3.5-turbo", [{"role": "user", "content": "hi"}])
    assert not result1.get("gptcache")
    assert result2.get("gptcache")

    fastrepl.LLMCache.disable()
    assert fastrepl.LLMCache.enabled() == False

    result3 = completion("gpt-3.5-turbo", [{"role": "user", "content": "hi"}])
    result4 = completion("gpt-3.5-turbo", [{"role": "user", "content": "hi"}])
    assert not result1.get("gptcache")
    assert not result1.get("gptcache")
