import fastrepl.repl as fastrepl


def fn_without_updatable():
    return "long text" * 100


def fn_with_updatable():
    return fastrepl.Updatable(key="test", value="long text" * 100)


def fn_with_updatable_repl():
    with fastrepl.REPL():
        fastrepl.Updatable(key="test", value="long text" * 100)


def test_fn_without_updatable(benchmark):
    benchmark(fn_without_updatable)


def test_fn_with_updatable(benchmark):
    benchmark(fn_with_updatable)


def test_fn_with_updatable_repl(benchmark):
    benchmark(fn_with_updatable_repl)
