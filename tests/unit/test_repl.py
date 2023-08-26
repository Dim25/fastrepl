import pytest

from fastrepl.loop import REPL
from fastrepl.polish import Updatable


class TestEval:
    def test_basic(self):
        def fn():
            Updatable(key="key_1", value="value_1"),
            Updatable(key="key_2", value="value_2"),

        with REPL() as controller:
            fn()


# TODO: current problem:
# We use ctx/key to target each Updatable
# and ctx: LocalContext and LocalContext is from inspect.stack, can not reproduce or target easily
# And we can not call Updatable().update() in local context.
# only thing we can do is just manipulate REPLContext(global).

# Actually this is user's convenience. In REPLContext, we don't care much

# Currently, we don't have idea of `loop`'s interface, so can not do this
