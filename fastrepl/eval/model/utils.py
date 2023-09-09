import random
from dataclasses import dataclass
from typing import Optional, Union, Literal, Iterable, List, Dict
from itertools import combinations

import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from fastrepl.utils import truncate
import fastrepl.llm as llm


def logit_bias_from(
    model: llm.SUPPORTED_MODELS, strings: Iterable[str]
) -> Dict[int, int]:
    def _get_token_id(s: str) -> int:
        ids = llm.tokenize(model, s)
        if len(ids) != 1:
            raise ValueError(f"{s!r} is not a single token in {model!r}")
        return ids[0]

    if model.startswith("command"):
        COHERE_MAX = 10
        return {_get_token_id(s): COHERE_MAX for s in strings}
    elif model.startswith("gpt"):
        OPENAI_MAX = 100
        return {_get_token_id(s): OPENAI_MAX for s in strings}
    else:
        return {}


@dataclass
class LabelMapping:
    token: str
    label: str
    description: str


def mappings_from_labels(
    labels: Dict[str, str], start=ord("A"), rg=random.Random(42)
) -> List[LabelMapping]:
    keys = rg.sample(list(labels.keys()), len(labels))
    return [
        LabelMapping(token=chr(start + i), label=label, description=labels[label])
        for i, label in enumerate(keys)
    ]


PositionDebiasStrategy: TypeAlias = Literal["shuffle", "consensus"]


# TODO: we can not be sure that every LLM has bias toward the first
def next_mappings_for_consensus(
    mappings: List[LabelMapping], result: Union[LabelMapping, str]
) -> Optional[List[LabelMapping]]:
    token = result.token if isinstance(result, LabelMapping) else result
    index = next(i for i, v in enumerate(mappings) if v.token == token)

    if len(mappings) % 2 == 1 and len(mappings) // 2 == index:
        return None

    ret = mappings[:]
    ret.reverse()
    return ret


def check_length_inbalance(texts: Iterable[str]) -> bool:
    RATIO = 0.5

    for a, b in combinations(texts, 2):
        (longer, shorter) = (a, b) if len(a) > len(b) else (b, a)
        if len(shorter) / len(longer) < RATIO:
            return True

    return False
