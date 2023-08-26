from fastrepl.polish import Updatable
from fastrepl.loop import REPL


class TestUpdatable:
    def test_value(self):
        with REPL():
            u = Updatable(key="key", value="value")
            assert u.value == "value"
