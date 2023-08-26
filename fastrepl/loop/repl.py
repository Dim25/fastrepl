import warnings
from typing import ClassVar, OrderedDict, List

from contextlib import ContextDecorator
import importlib.metadata

from rich.progress import Progress
from multiprocessing.pool import ThreadPool

from fastrepl.eval import Evaluator
from fastrepl.utils import LocalContext, getenv, get_cuid


NUM_THREADS = getenv("NUM_THREADS", 8)
DEFAULT_INFO = {"fastrepl": importlib.metadata.version("fastrepl")}


class REPLController:
    __slots__ = ("id", "info", "_evaluator", "_display")

    def __init__(self):
        self.id = get_cuid()
        self.info = DEFAULT_INFO

    def set_evaluator(self, evaluator: Evaluator):
        self._evaluator = evaluator

    def eval(self, inputs: List[str]) -> List[str]:
        ret = []
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=len(inputs))

            with ThreadPool(NUM_THREADS) as pool:
                for result in pool.imap(self._evaluator.run, inputs):
                    ret.append(result)
                    progress.update(task, advance=1, refresh=True)
        return ret


class REPLContext:
    latest_id: ClassVar[str] = get_cuid()
    head_id: ClassVar[str] = latest_id
    ctx_key_values: ClassVar[dict[LocalContext, dict[str, OrderedDict[str, str]]]] = {}

    @staticmethod
    def reset():
        REPLContext.ctx_key_values = {}

    @staticmethod
    def trace(ctx: LocalContext, key: str, value: str):
        if ctx not in REPLContext.ctx_key_values:
            REPLContext.ctx_key_values[ctx] = {}
        if key not in REPLContext.ctx_key_values[ctx]:
            REPLContext.ctx_key_values[ctx][key] = OrderedDict()
        REPLContext.ctx_key_values[ctx][key][REPLContext.latest_id] = value

    # TODO: This unlikely the final API, we might use Runner to update the value.
    @staticmethod
    def update(target_ctx: LocalContext, target_key: str, new_value: str):
        for ctx, key_values in REPLContext.ctx_key_values.items():
            for key, values in key_values.items():
                if ctx == target_ctx and key == target_key:
                    values[REPLContext.latest_id] = new_value
                else:
                    # TODO: Here we just copy the previous value
                    previous = next(reversed(values.values()))
                    values[REPLContext.latest_id] = previous

    @staticmethod
    def get_current_value(ctx: LocalContext, key: str) -> str:
        history = REPLContext.ctx_key_values[ctx][key]
        ret = history.get(REPLContext.head_id, None)
        if ret is None:
            k, v = next(reversed(history.items()))
            warnings.warn(
                f"when retrieving {key!r}, {REPLContext.head_id!r} not found. Using {k!r} instead",
                UserWarning,
            )
            return v
        return ret


class REPL(ContextDecorator):
    __slots__ = "controller"

    def __init__(self):
        self.controller = REPLController()

    def __enter__(self) -> REPLController:
        return self.controller

    def __exit__(self, *args):
        REPLContext.reset()
        self.controller = None
