import inspect

from fastrepl.utils import LocalContext
from fastrepl.repl.context import REPLContext


class Updatable:
    def __init__(
        self,
        key: str,
        value: str,
        what="this is updatable value",
        how="be creative while maintaining the original meaning",
    ):
        self._key, self._value, self.what, self.how = key, value, what, how

        from fastrepl.repl import IS_REPL

        if IS_REPL.get():
            self._ctx = LocalContext(inspect.stack()[1])
            REPLContext.trace(self._ctx, self._key, value)

    @property
    def value(self) -> str:
        from fastrepl.repl import IS_REPL

        if IS_REPL.get():
            return REPLContext.get_current(self._ctx, self._key)
        return self._value
