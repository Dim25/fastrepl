import pytest
import fastrepl.repl as fastrepl


@pytest.fixture
def enable_interactive(monkeypatch):
    monkeypatch.setenv("FASTREPL_INTERACTIVE", "1")


@pytest.fixture
def disable_interactive(monkeypatch):
    monkeypatch.setenv("FASTREPL_INTERACTIVE", "0")


def fn_without_updatable():
    return "long text" * 100


def fn_with_updatable():
    return fastrepl.Updatable(key="test", value="long text" * 100)


def fn_with_updatable_repl():
    return fastrepl.Updatable(key="test", value="long text" * 100)


def test_fn_without_updatable(benchmark):
    benchmark(fn_without_updatable)


def test_fn_with_updatable(benchmark, disable_interactive):
    benchmark(fn_with_updatable)


def test_fn_with_updatable_repl(benchmark, enable_interactive):
    benchmark(fn_with_updatable_repl)
