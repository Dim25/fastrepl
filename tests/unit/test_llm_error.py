import pytest
import openai.error
import litellm
import litellm.gpt_cache


from fastrepl.llm import (
    handle_llm_exception,
    RetryConstantException,
    RetryExpoException,
    completion,
)


class TestHandleLLMException:
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


class TestContextFallback:
    def test_short(self, monkeypatch):
        def mock(**kwargs):
            if kwargs.get("model") == "gpt-3.5-turbo":
                raise litellm.exceptions.ContextWindowExceededError("", "", "")
            elif kwargs.get("model") == "gpt-3.5-turbo-16k":
                return {"choices": [{"finish_reason": "stop"}]}
            else:
                raise NotImplementedError

        monkeypatch.setattr(litellm.gpt_cache, "completion", mock)

        completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "8k tokens"}],
        )

    def test_long(self, monkeypatch):
        def mock(**kwargs):
            if kwargs.get("model") == "gpt-3.5-turbo":
                raise litellm.exceptions.ContextWindowExceededError("", "", "")
            elif kwargs.get("model") == "gpt-3.5-turbo-16k":
                raise litellm.exceptions.ContextWindowExceededError("", "", "")
            else:
                raise NotImplementedError

        monkeypatch.setattr(litellm.gpt_cache, "completion", mock)

        with pytest.raises(litellm.exceptions.ContextWindowExceededError):
            completion(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": "24k tokens"}],
            )
