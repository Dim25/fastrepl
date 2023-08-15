from fastrepl.polish import Updatable


def fn_without_updatable():
    return "long text" * 100


def fn_with_updatable():
    return Updatable(key="test", value="long text" * 100)


def test_fn_without_updatable(benchmark):
    benchmark(fn_without_updatable)


def test_fn_with_updatable(benchmark):
    benchmark(fn_with_updatable)
