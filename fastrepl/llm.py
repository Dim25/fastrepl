from typing import Literal, List, Dict
import os
import functools

import backoff
import openai.error

import fastrepl
from fastrepl.utils import getenv, pprint


from gptcache import cache
from gptcache.manager import get_data_manager


def cache_enable_func(*args, **kwargs):
    return fastrepl.LLMCache.enabled()


dir_name, _ = os.path.split(os.path.abspath(__file__))
cache.init(
    cache_enable_func=cache_enable_func,
    data_manager=get_data_manager(
        data_path=f"{dir_name}/.fastrepl.cache", max_size=1000
    ),
)

import litellm
from litellm import ModelResponse
from litellm.cache import completion as litellm_completion

litellm.telemetry = False  # pragma: no cover
litellm.caching = True  # pragma: no cover
litellm.caching_with_models = True  # pragma: no cover


# TODO: This logging should be added to some global context and rendered in Live display
def check_retryable_and_log(exp: Exception) -> bool:
    if isinstance(exp, openai.error.ServiceUnavailableError):
        pprint("[bright_red]Service Unavailable[/bright_red]")
    elif isinstance(exp, openai.error.APIError):
        pprint("[bright_red]API Error[/bright_red]")
    elif isinstance(exp, openai.error.RateLimitError):
        pprint("[bright_red]Rate Limit Error[/bright_red]")
    elif isinstance(exp, openai.error.APIConnectionError):
        pprint("[bright_red]API Connection Error[/bright_red]")
    elif isinstance(exp, openai.error.Timeout):
        pprint("[bright_red]Timeout[/bright_red]")
    else:
        return False
    return True


class RetryableException(Exception):
    def __init__(self):
        super(Exception, self).__init__("retryable exception from fastrepl")


SUPPORTED_MODELS = Literal[  # pragma: no cover
    # https://docs.litellm.ai/docs/completion/supported#openai-chat-completion-models
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    # https://docs.litellm.ai/docs/completion/supported#ai21-models
    "j2-ultra",
    # https://docs.litellm.ai/docs/completion/supported#cohere-models
    "command-nightly",
    # https://docs.litellm.ai/docs/completion/supported#together-ai-models
    "togethercomputer/llama-2-70b-chat",
]


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(RetryableException),
    max_value=60,
    factor=1.5,
)
def completion(
    model: SUPPORTED_MODELS,
    messages: List[Dict[str, str]],
    temperature: float = 0,
    logit_bias: Dict[int, int] = {},
    max_tokens: int = 100,
) -> ModelResponse:
    try:
        result = litellm_completion(  # pragma: no cover
            model=model,
            messages=messages,
            temperature=temperature,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            force_timeout=30,
        )
        return result
    except Exception as e:
        if check_retryable_and_log(e):
            raise RetryableException() from e
        else:
            raise e


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
