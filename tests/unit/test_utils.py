import pytest

from fastrepl.utils import HistoryDict, ensure


class TestDataStructure:
    def test_ordered_set(self):
        from fastrepl.utils import OrderedSet

        o = OrderedSet[str]()
        assert len(o) == 0
        assert o.keys() == []

        # TODO: Why no type error
        o.add(1)
        assert len(o) == 1
        assert o.keys() == [1]

        o = OrderedSet[str]()
        assert len(o) == 0
        assert o.keys() == []

        o.add("a")
        o.add("b")
        assert len(o) == 2
        assert o.keys() == ["a", "b"]

        assert "a" in o
        assert "b" in o
        assert "c" not in o

        with pytest.raises(NotImplementedError):
            o == []

    def test_history_dict(self):
        with pytest.raises(NotImplementedError):
            hd = HistoryDict[str](initial="initial")
            assert hd is not None


class TestDecorator:
    def test_ensure(self):
        x = 0

        @ensure(lambda: x == 0)
        def fn1():
            return True

        @ensure(lambda: x != 0)
        def fn2():
            return True

        assert fn1()
        with pytest.raises(AssertionError):
            assert fn2()
