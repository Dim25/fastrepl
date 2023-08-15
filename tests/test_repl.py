import pytest

from fastrepl import REPL
from fastrepl.polish import Updatable


class TestEval:
    def test_basic(self):
        def fn():
            Updatable(key="key_1", value="value_1"),
            Updatable(key="key_2", value="value_2"),

        with REPL() as controller:
            fn()
            with pytest.raises(NotImplementedError):
                controller.add_eval()
