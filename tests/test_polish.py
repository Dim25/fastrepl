from fastrepl.polish import Updatable


class TestUpdatable:
    def test_value(self):
        u = Updatable(key="key", value="value")
        assert str(u) == "value"
