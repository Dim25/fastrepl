import inspect

from fastrepl.context import AnalyzeContext, REPLContext, LocalContext

DEFAULT_WHAT = "this is updatable value"
DEFAULT_HOW = "be creative while maintaining the original meaning"


class Updatable:
    __slots__ = ("_ctx", "_key", "what", "how")

    def __init__(self, key: str, value: str, what=DEFAULT_WHAT, how=DEFAULT_HOW):
        self._key, self.what, self.how = key, what, how

        self._ctx = LocalContext(inspect.stack()[1])

        # TODO: We should not do this. Need to introduce contextvar?
        AnalyzeContext.trace(self._ctx, self._key, value)
        REPLContext.trace(self._ctx, self._key, value)

    def __repr__(self) -> str:
        return f"Updatable({self._ctx!r}, {self._key!r})"

    @property
    def value(self) -> str:
        return REPLContext.get_current_value(self._ctx, self._key)
