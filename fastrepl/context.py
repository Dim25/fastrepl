import os
from inspect import FrameInfo
from typing import ClassVar, List, OrderedDict, Any

from fastrepl.utils import OrderedSet


class LocalContext:
    __slots__ = ("_filename", "_function")

    _filename: str
    _function: str

    def __init__(self, frame: FrameInfo) -> None:
        self._filename, self._function = (
            os.path.basename(frame.filename),
            frame.function,
        )

    def __str__(self) -> str:
        return f"{self._filename}:{self._function}"

    def __repr__(self) -> str:
        return f"VariableContext({self._filename!r}, {self._function!r})"

    def __hash__(self) -> int:
        return hash((self._filename, self._function))

    def __eq__(self, v: object) -> bool:
        if not isinstance(v, LocalContext):
            return False
        return self._filename == v._filename and self._function == v._function

    @property
    def filename(self):
        return self._filename

    @property
    def function(self):
        return self._function


class GlobalContext:
    # for Analyze
    nth_run: ClassVar[int] = 0
    run_ctx_keys: ClassVar[List[OrderedDict[LocalContext, OrderedSet[str]]]] = []
    # for REPL
    ctx_key_values: ClassVar[Any]

    @staticmethod
    def reset():
        GlobalContext.nth_run = 0
        GlobalContext.run_ctx_keys = []
        GlobalContext.ctx_key_values = None  # TODO

    @staticmethod
    def reset_analyze():
        GlobalContext.nth_run = 0
        GlobalContext.run_ctx_keys = []

    @staticmethod
    def reset_repl():
        GlobalContext.nth_run = 0
        GlobalContext.run_ctx_keys = []

    @staticmethod
    def trace(ctx: LocalContext, key: str, value: str):
        # Filling `run_ctx_keys`
        if len(GlobalContext.run_ctx_keys) == GlobalContext.nth_run:
            GlobalContext.run_ctx_keys.append(OrderedDict())
        if ctx not in GlobalContext.run_ctx_keys[GlobalContext.nth_run]:
            GlobalContext.run_ctx_keys[GlobalContext.nth_run][ctx] = OrderedSet()

        keys = GlobalContext.run_ctx_keys[GlobalContext.nth_run][ctx]
        if key in keys:
            msg = f"in {GlobalContext.nth_run}th run, key={key} already exists in ctx={ctx}. Maybe you forgot to call `next_run`?"
            raise ValueError(msg)
        keys.add(key)

    @staticmethod
    def next_run():
        if len(GlobalContext.run_ctx_keys) <= GlobalContext.nth_run:
            msg = "you need to call `trace` at least once before calling `next_run`"
            raise ValueError(msg)

        GlobalContext.nth_run += 1
