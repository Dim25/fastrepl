import pytest

import openai.error
import litellm.exceptions

from fastrepl.llm import (
    handle_llm_exception,
    RetryConstantException,
    RetryExpoException,
)


class TestOpenAI:
    @pytest.mark.parametrize(
        "exception",
        [
            openai.error.APIError(),
            openai.error.TryAgain(),
            openai.error.Timeout(),
            openai.error.ServiceUnavailableError(),
        ],
    )
    def test_constant(self, exception):
        with pytest.raises(RetryConstantException):
            handle_llm_exception(exception)

    @pytest.mark.parametrize(
        "exception",
        [
            openai.error.RateLimitError(),
        ],
    )
    def test_expo(self, exception):
        with pytest.raises(RetryExpoException):
            handle_llm_exception(exception)

    @pytest.mark.parametrize(
        "exception",
        [
            openai.error.APIConnectionError(""),
            openai.error.InvalidRequestError("", ""),
            openai.error.AuthenticationError(),
            openai.error.PermissionError(),
            openai.error.InvalidAPIType(),
            openai.error.SignatureVerificationError("", ""),
        ],
    )
    def test_no_retry(self, exception):
        with pytest.raises(type(exception)):
            handle_llm_exception(exception)


class TestLiteLLM:
    @pytest.mark.parametrize(
        "exception",
        [
            litellm.exceptions.ServiceUnavailableError("", ""),
        ],
    )
    def test_constant(self, exception):
        with pytest.raises(RetryConstantException):
            handle_llm_exception(exception)

    @pytest.mark.parametrize(
        "exception",
        [
            litellm.exceptions.RateLimitError("", ""),
        ],
    )
    def test_expo(self, exception):
        with pytest.raises(RetryExpoException):
            handle_llm_exception(exception)

    @pytest.mark.parametrize(
        "exception",
        [
            litellm.exceptions.AuthenticationError("", ""),
            litellm.exceptions.InvalidRequestError("", "", ""),
        ],
    )
    def test_no_retry(self, exception):
        with pytest.raises(type(exception)):
            handle_llm_exception(exception)
