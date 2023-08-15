from typing import TypeVar, Iterable, Tuple
from itertools import tee

PairwiseItem = TypeVar("PairwiseItem")


def pairwise(
    iterable: Iterable[PairwiseItem],
) -> Iterable[Tuple[PairwiseItem, PairwiseItem]]:  # pragma: no cover
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
