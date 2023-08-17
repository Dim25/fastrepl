from typing import overload, get_args

from fastrepl.eval.metric.huggingface import (
    HUGGINGFACE_BUILTIN_METRICS,
    HUGGINGFACE_FASTREPL_METRICS,
    HuggingfaceMetric,
)


@overload
def load_metric(name: HUGGINGFACE_BUILTIN_METRICS) -> HuggingfaceMetric:
    ...


@overload
def load_metric(name: HUGGINGFACE_FASTREPL_METRICS) -> HuggingfaceMetric:
    ...


def load_metric(name: str) -> HuggingfaceMetric:
    if name in get_args(HUGGINGFACE_BUILTIN_METRICS):
        return HuggingfaceMetric(name)
    elif name in get_args(HUGGINGFACE_FASTREPL_METRICS):
        return HuggingfaceMetric(name)
    else:
        raise NotImplementedError
