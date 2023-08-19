import warnings
from typing import ClassVar, List, OrderedDict

from fastrepl.utils import OrderedSet
from fastrepl.context.local import LocalContext


class AnalyzeContext:
    nth_run: ClassVar[int] = 0
    run_ctx_keys: ClassVar[List[OrderedDict[LocalContext, OrderedSet[str]]]] = []

    @staticmethod
    def reset():
        AnalyzeContext.nth_run = 0
        AnalyzeContext.run_ctx_keys = []

    # TODO: rename, register?
    @staticmethod
    def trace(ctx: LocalContext, key: str, value: str):
        while len(AnalyzeContext.run_ctx_keys) <= AnalyzeContext.nth_run:
            AnalyzeContext.run_ctx_keys.append(OrderedDict())

        if ctx not in AnalyzeContext.run_ctx_keys[AnalyzeContext.nth_run]:
            AnalyzeContext.run_ctx_keys[AnalyzeContext.nth_run][ctx] = OrderedSet()

        keys = AnalyzeContext.run_ctx_keys[AnalyzeContext.nth_run][ctx]
        if key in keys:
            warnings.warn(
                f"{key!r} already exists in {ctx!r}. Maybe you forgot to call 'next_run()'?",
                UserWarning,
            )
        keys.add(key)

    @staticmethod
    def next_run():
        if len(AnalyzeContext.run_ctx_keys) == AnalyzeContext.nth_run:
            warnings.warn(
                f"trace() should be called at least once before next_run()",
                UserWarning,
            )
        AnalyzeContext.nth_run += 1
