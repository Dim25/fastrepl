from typing import Literal, List, Dict
import json
import functools

import backoff
import litellm
from litellm import completion as litellm_completion, ModelResponse

litellm.telemetry = False  # pragma: no cover
litellm.caching = False  # pragma: no cover
litellm.caching_with_models = False  # pragma: no cover

import fastrepl
from fastrepl.utils import getenv
from fastrepl.run.error import check_retryable_and_log, RetryableException

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
    if fastrepl.cache is not None:
        hit = fastrepl.cache.lookup(model, prompt=json.dumps(messages))
        if hit is not None:
            ret = json.loads(hit)
            ret["_fastrepl_cached"] = True
            return ret  # TODO: This is not ModelResponse anymore

    try:
        result = litellm_completion(  # pragma: no cover
            model,
            messages,
            temperature=temperature,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            force_timeout=30,
        )
        if fastrepl.cache is not None:
            fastrepl.cache.update(
                model, prompt=json.dumps(messages), response=json.dumps(result)
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
