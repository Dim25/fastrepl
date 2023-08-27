import pytest
import warnings

from fastrepl.utils import LocalContext
from fastrepl.repl.context import REPLContext
from fastrepl.errors import InvalidStatusError


@pytest.fixture(autouse=True)
def set_context_and_teardown():
    REPLContext.reset()
    yield
    REPLContext.reset()


class MockLocalContext(LocalContext):
    def __init__(self, filename: str, function: str) -> None:
        self._filename = filename
        self._function = function


class TestREPLContext:
    def test_initial_status(self):
        status = REPLContext._status
        history = REPLContext._history
        assert [status] == history

    def test_reset(self):
        status = REPLContext._status
        REPLContext.reset()
        assert REPLContext._status != status

    def test_current_value_without_update(self):
        REPLContext.trace(MockLocalContext("a", "b"), "key", "value")

        REPLContext.get_current(MockLocalContext("a", "b"), "key") == "value"
        with pytest.raises(KeyError):
            REPLContext.get_current(MockLocalContext("a", "c"), "key")

    def test_current_value_with_update(self):
        REPLContext.trace(MockLocalContext("a", "b"), "key", "value1")

        REPLContext.update([("key", "value2")])
        REPLContext.get_current(MockLocalContext("a", "b"), "key") == "value2"

    def test_set_status_invalid(self):
        current = REPLContext._status
        REPLContext.set_status(current)

        with pytest.raises(InvalidStatusError):
            REPLContext.set_status("status")

    def test_current_value_set_status(self):
        initial_status = REPLContext._status

        REPLContext.trace(MockLocalContext("a", "b"), "key1", "value1")
        REPLContext.trace(MockLocalContext("a", "b"), "key2", "value2")

        REPLContext.get_current(MockLocalContext("a", "b"), "key1") == "value1"
        REPLContext.get_current(MockLocalContext("a", "b"), "key2") == "value2"

        REPLContext.update([("key1", "value3")])
        REPLContext.get_current(MockLocalContext("a", "b"), "key1") == "value3"
        REPLContext.get_current(MockLocalContext("a", "b"), "key2") == "value2"

        REPLContext.set_status(initial_status)
        REPLContext.get_current(MockLocalContext("a", "b"), "key1") == "value1"
        REPLContext.get_current(MockLocalContext("a", "b"), "key2") == "value2"
