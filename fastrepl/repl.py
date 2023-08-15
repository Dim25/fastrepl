from contextlib import ContextDecorator

from fastrepl.context import GlobalContext


class REPLController:
    @staticmethod
    def add_eval():
        raise NotImplementedError


# TODO
class REPL(ContextDecorator):
    def __init__(self):
        self.controller = REPLController()

    def __enter__(self) -> REPLController:
        return self.controller

    def __exit__(self, *args):
        GlobalContext.reset_repl()
