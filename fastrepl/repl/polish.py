import inspect

from fastrepl.utils import LocalContext, getenv
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

        if getenv("FASTREPL_INTERACTIVE", 0) > 0:
            print("registering updatable", key, value)
            self._ctx = LocalContext(inspect.stack()[1])
            REPLContext.trace(self._ctx, self._key, value)

    @property
    def value(self) -> str:
        if getenv("FASTREPL_INTERACTIVE", 0) > 0:
            return REPLContext.get_current(self._ctx, self._key)
        return self._value
