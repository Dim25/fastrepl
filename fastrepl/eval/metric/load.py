import functools
from typing import overload, get_args


from fastrepl.eval.base import BaseMetaEvalNode
from fastrepl.eval.metric.huggingface import (
    HUGGINGFACE_BUILTIN_METRICS,
    HUGGINGFACE_FASTREPL_METRICS,
    HuggingfaceMetric,
)
from fastrepl.eval.metric.sas import (
    SENTENCE_ANSWER_SIMILARITY_METRICS,
    SemanticAnswerSimilarityMetric,
)


@overload
def load_metric(name: HUGGINGFACE_BUILTIN_METRICS, **kwargs) -> HuggingfaceMetric:
    ...


@overload
def load_metric(name: HUGGINGFACE_FASTREPL_METRICS, **kwargs) -> HuggingfaceMetric:
    ...


@overload
def load_metric(
    name: SENTENCE_ANSWER_SIMILARITY_METRICS, **kwargs
) -> SemanticAnswerSimilarityMetric:
    ...


@functools.lru_cache()
def load_metric(name: str, **kwargs) -> BaseMetaEvalNode:
    if name in get_args(HUGGINGFACE_BUILTIN_METRICS):
        return HuggingfaceMetric(name)
    elif name in get_args(HUGGINGFACE_FASTREPL_METRICS):
        return HuggingfaceMetric(name)
    elif name in get_args(SENTENCE_ANSWER_SIMILARITY_METRICS):
        model_name_or_path, use_gpu = (
            kwargs.pop("model_name_or_path"),
            kwargs.pop("use_gpu", False),
        )
        return SemanticAnswerSimilarityMetric(model_name_or_path, use_gpu)
    else:
        raise NotImplementedError
