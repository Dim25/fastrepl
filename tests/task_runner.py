from fastrepl.loop import ClassificationRunner


class TestClassificationRunner:
    def test_basic(self):
        assert ClassificationRunner() is not None
