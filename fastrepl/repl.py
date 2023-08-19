from typing import Any
from contextlib import ContextDecorator

from fastrepl.utils import get_cuid
from fastrepl.context import REPLContext


DEFAULT_INFO = {
    # TODO: Get version info from somewhere, same for docs
    "fastrepl": "0.0.1",
}


class REPLController:
    __slots__ = ("_id", "_info", "_evaluators", "_polishers")

    def __init__(self):
        self._id = get_cuid()
        self._info = DEFAULT_INFO

    def set_evaluators(self, auto=True, list=[]):
        self._evaluators = list

    def set_polishers(self, auto=True, list=[]):
        self._polishers = list

    def eval(self) -> dict:
        # TODO
        return {
            "f1": 0.1,
            "accuracy": 0.2,
            "model_graded": 0.7,
            "mean_reciprocal_rank": 0.3,
            "mean_average_precision": 0.4,
        }

    def polish(self):
        pass

    # TODO: Additionals is for non-e2e evals. For ex, `evaluate_source_nodes` in llama_index
    # I think we also need some key here. like where does additional metric is for. ex, retriever
    def add_pair(self, input: Any, output: Any, additionals=[]):
        ...

    def build_report(self, save=False, path="./fastrepl.json") -> dict:
        return {
            "id": self.id,
            "info": self.info,
        }

    @staticmethod
    def load_report(path="./fastrepl.json"):
        pass

    @property
    def id(self) -> str:
        return self._id

    @property
    def info(self) -> dict:
        if len(self._evaluators) == 0:
            raise ValueError("evaluators are not set")
        if len(self._polishers) == 0:
            raise ValueError("polishers are not set")
        return self._info


class REPL(ContextDecorator):
    __slots__ = "controller"

    def __init__(self):
        self.controller = REPLController()

    def __enter__(self) -> REPLController:
        return self.controller

    def __exit__(self, *args):
        REPLContext.reset()
        self.controller = None
