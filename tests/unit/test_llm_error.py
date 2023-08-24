import pytest
import openai.error
import litellm.exceptions

from fastrepl.run.error import check_retryable_and_log, RetryableException


class TesthandleLLMError:
    @pytest.mark.parametrize(
        "exception",
        [
            openai.error.ServiceUnavailableError(),
            openai.error.APIError(),
            openai.error.RateLimitError(),
            openai.error.APIConnectionError(""),
            openai.error.Timeout(),
            # TODO: more exception from litellm
            litellm.exceptions.ServiceUnavailableError("", ""),
            litellm.exceptions.RateLimitError("", ""),
        ],
    )
    def test_retryable_basic(self, exception):
        assert check_retryable_and_log(exception)

    @pytest.mark.parametrize(
        "exception",
        [
            openai.error.AuthenticationError(),
            openai.error.InvalidRequestError("", ""),
            #
            litellm.exceptions.AuthenticationError("", ""),
            litellm.exceptions.InvalidRequestError("", "", ""),
        ],
    )
    def test_not_retryable_basic(self, exception):
        assert not check_retryable_and_log(exception)
