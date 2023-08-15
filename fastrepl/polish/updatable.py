import inspect
from typing import List

from fastrepl.context import GlobalContext, LocalContext


class Updatable:
    values: List[str]
    feedbacks: List[str]

    _render_template: str = """### Description of this value:
{what}

How to update this value:
{how}

{history}
"""

    def __init__(self, key: str, value: str) -> None:
        frame = inspect.stack()[1]
        ctx = LocalContext(frame)

        GlobalContext.trace(ctx, key, value)
        self.values = [value]  # TODO
        self.feedbacks = []  # TODO

    def __str__(self) -> str:
        return self.values[-1]

    def _what(self) -> str:
        raise NotImplementedError

    def _how(self) -> str:
        raise NotImplementedError

    def _render(self) -> str:  # TODO
        history = ""
        return self._render_template.format(self._what(), self._how(), history)
