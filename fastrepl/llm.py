from typing import Literal, List, Dict, Any
import os
import functools
import warnings

import backoff
import openai.error

import fastrepl
from fastrepl.utils import getenv, debug

from gptcache import cache
from gptcache.processor.pre import last_content_without_prompt
from gptcache.manager import get_data_manager


def cache_enable_func(*args, **kwargs):
    return fastrepl.LLMCache.enabled()


def pre_cache_func(data: Dict[str, Any], **params: Dict[str, Any]) -> Any:
    last_content_without_prompt_val = last_content_without_prompt(data, **params)
    cache_key = last_content_without_prompt_val + data["model"]
    return cache_key


dir_name, _ = os.path.split(os.path.abspath(__file__))
cache.init(
    cache_enable_func=cache_enable_func,
    pre_func=pre_cache_func,
    data_manager=get_data_manager(
        data_path=f"{dir_name}/.fastrepl.cache", max_size=1000
    ),
)

import litellm
import litellm.exceptions
import litellm.gpt_cache
from litellm import ModelResponse

litellm.telemetry = False  # pragma: no cover


class RetryConstantException(Exception):
    pass


class RetryExpoException(Exception):
    pass


def handle_llm_exception(e: Exception):
    if isinstance(
        e,
        (
            openai.error.APIError,
            openai.error.TryAgain,
            openai.error.Timeout,
            openai.error.ServiceUnavailableError,
        ),
    ):
        raise RetryConstantException from e
    elif isinstance(e, openai.error.RateLimitError):
        raise RetryExpoException from e
    elif isinstance(
        e,
        (
            openai.error.APIConnectionError,
            openai.error.InvalidRequestError,  # NOTE: ContextWindowExceededError will be catched outside
            openai.error.AuthenticationError,
            openai.error.PermissionError,
            openai.error.InvalidAPIType,
            openai.error.SignatureVerificationError,
        ),
    ):
        raise e
    else:
        warnings.warn(f"got unknown exception: {type(e)}")
        raise e


SUPPORTED_MODELS = Literal[  # pragma: no cover
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "j2-ultra",
    "command-nightly",
    "togethercomputer/llama-2-70b-chat",
]


@backoff.on_exception(
    wait_gen=backoff.constant,
    exception=(RetryConstantException),
    max_tries=3,
    interval=3,
)
@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(RetryExpoException),
    jitter=backoff.full_jitter,
    max_value=100,
    factor=1.5,
)
def completion(
    model: SUPPORTED_MODELS,
    messages: List[Dict[str, str]],
    temperature: float = 0,
    logit_bias: Dict[int, int] = {},
    max_tokens: int = 200,
) -> ModelResponse:
    # TODO: this should be done in eval side
    debug(messages, before=f"completion({model})")

    LONGER_CONTEXT_MAPPING = {  # pragma: no cover
        "gpt-3.5-turbo": "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0613": "gpt-3.5-turbo-16k-0613",
        "gpt-4": "gpt-4-32k",
        "gpt-4-0314": "gpt-4-32k-0314",
        "gpt-4-0613": "gpt-4-32k-0613",
    }

    def _completion(fallback=None):
        try:
            maybe_fallback = fallback if fallback is not None else model

            result = litellm.gpt_cache.completion(  # pragma: no cover
                model=maybe_fallback,
                messages=messages,
                temperature=temperature,
                logit_bias=logit_bias,
                max_tokens=max_tokens,
                force_timeout=20,
            )

            if result["choices"][0]["finish_reason"] == "length":
                warnings.warn(f"{maybe_fallback} completion truncated due to length")

            return result
        except Exception as e:
            handle_llm_exception(e)

    try:
        return _completion()
    except litellm.exceptions.ContextWindowExceededError as e:
        if LONGER_CONTEXT_MAPPING.get(model) is None:
            raise e
        return _completion(fallback=LONGER_CONTEXT_MAPPING[model])


@functools.lru_cache(maxsize=None)
def tokenize(model: SUPPORTED_MODELS, text: str) -> List[int]:
    if model == "command-nightly":
        import cohere

        co = cohere.Client(getenv("COHERE_API_KEY", ""))
        response = co.tokenize(text=text, model="command")
        return response.tokens
    elif model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode(text)

    # https://docs.ai21.com/reference/tokenize-ref
    raise NotImplementedError(f"tokenize not implemented for {model!r}")
