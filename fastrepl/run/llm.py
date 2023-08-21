from typing import Literal, List, Dict
import functools

import litellm
from litellm import completion as litellm_completion

litellm.telemetry = False  # pragma: no cover
litellm.caching_with_models = True  # pragma: no cover

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


def completion(
    model: SUPPORTED_MODELS,
    messages: List[Dict[str, str]],
    temperature: float = 0,
    logit_bias: Dict[int, int] = {},
    max_tokens: int = 100,
):
    return litellm_completion(  # pragma: no cover
        model,
        messages,
        temperature=temperature,
        logit_bias=logit_bias,
        max_tokens=max_tokens,
    )


from fastrepl.utils import getenv


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
