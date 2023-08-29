import random
from dataclasses import dataclass
from typing import Set, List, Dict


from fastrepl.llm import SUPPORTED_MODELS, tokenize


def logit_bias_from_labels(model: SUPPORTED_MODELS, labels: Set[str]) -> Dict[int, int]:
    def get_token_id(label: str) -> int:
        ids = tokenize(model, label)
        if len(ids) != 1:
            raise ValueError(f"{label!r} is not a single token in {model!r}")
        return ids[0]

    if model == "command-nightly":
        COHERE_MAX = 10
        return {get_token_id(k): COHERE_MAX for k in labels}
    elif model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]:
        OPENAI_MAX = 100
        return {get_token_id(k): OPENAI_MAX for k in labels}
    else:
        return {}


@dataclass
class LabelMapping:
    token: str
    label: str
    description: str


def mapping_from_labels(
    labels: Dict[str, str], start=ord("A"), rg=random.Random(42)
) -> List[LabelMapping]:
    keys = rg.sample(list(labels.keys()), len(labels))
    return [
        LabelMapping(token=chr(start + i), label=label, description=labels[label])
        for i, label in enumerate(keys)
    ]
