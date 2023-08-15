import pytest
import inspect

from fastrepl.context import GlobalContext, LocalContext


class TestLocalContext:
    class TestStack:
        def test_depth_1(self):
            assert inspect.stack()[0].filename == __file__
            assert inspect.stack()[0].function == "test_depth_1"

        def test_depth_2_fn(self):
            def a():
                return inspect.stack()

            assert a()[0].filename == __file__
            assert a()[0].function == "a"
            assert a()[1].filename == __file__
            assert a()[1].function == "test_depth_2_fn"

        def test_depth_2_class(self):
            class B:
                def __init__(self):
                    self.stack = inspect.stack()

            b = B()
            assert b.stack[0].filename == __file__
            assert b.stack[0].function == "__init__"
            assert b.stack[1].filename == __file__
            assert b.stack[1].function == "test_depth_2_class"

        def test_depth3(self):
            class B:
                def __init__(self):
                    self.stack = inspect.stack()

            def c():
                b = B()
                return b.stack

            assert c()[0].filename == __file__
            assert c()[0].function == "__init__"
            assert c()[1].filename == __file__
            assert c()[1].function == "c"
            assert c()[2].filename == __file__

    def test_one_frame(self):
        def fn():
            frame = inspect.stack()[1]
            return LocalContext(frame)

        c0 = fn()

        assert c0 is c0
        assert c0 == c0
        assert c0.__hash__() == c0.__hash__()
        d = {}
        d[c0], d[c0] = 1, 2
        assert len(d) == 1
        assert d[c0] == 2

        assert c0.function == "test_one_frame"

    def test_two_frame(self):
        def fn():
            frame0, frame1 = inspect.stack()[0], inspect.stack()[1]
            return LocalContext(frame0), LocalContext(frame1)

        c0, c1 = fn()

        assert c0 is not c1
        assert c0 != c1
        assert c0.__hash__() != c1.__hash__()
        d = {}
        d[c0], d[c1] = 1, 2
        assert len(d) == 2

        assert c0.function == "fn"
        assert c1.function == "test_two_frame"

    def test_eq(self):
        def fn():
            frame0, frame1 = inspect.stack()[0], inspect.stack()[0]
            return LocalContext(frame0), LocalContext(frame1)

        c0, c1 = fn()
        assert c0 is not c1
        assert c0 == c1

    def test_neq(self):
        def fn():
            frame0, frame1 = inspect.stack()[0], inspect.stack()[1]
            return LocalContext(frame0), LocalContext(frame1)

        c0, c1 = fn()
        assert c0 is not c1
        assert c0 != c1

    def test_hash_eq(self):
        def fn():
            frame0, frame1 = inspect.stack()[0], inspect.stack()[0]
            return LocalContext(frame0), LocalContext(frame1)

        c0, c1 = fn()
        d = {}
        d[c0], d[c1] = 1, 2
        assert len(d) == 1
        assert d[c0] == 2
        assert d[c1] == 2

    def test_hash_neq(self):
        def fn():
            frame0, frame1 = inspect.stack()[0], inspect.stack()[1]
            return LocalContext(frame0), LocalContext(frame1)

        c0, c1 = fn()
        d = {}
        d[c0], d[c1] = 1, 2
        assert len(d) == 2
        assert d[c0] == 1
        assert d[c1] == 2


class TestGlobalContext:
    class TestTrace:
        class TestGlobalContextTrace:
            def test_empty(self):
                GlobalContext.reset()
                assert GlobalContext.nth_run == 0
                assert len(GlobalContext.run_ctx_keys) == 0

            def test_single_run_ctx(self):
                GlobalContext.reset()
                GlobalContext.trace(ctx="ctx1", key="key1", value="value1")
                assert len(GlobalContext.run_ctx_keys) == 1

                nth_run = GlobalContext.nth_run
                assert GlobalContext.run_ctx_keys[nth_run]["ctx1"].keys() == ["key1"]

            def test_single_run_ctx_multiple_keys(self):
                GlobalContext.reset()

                GlobalContext.trace(ctx="ctx1", key="key1", value="value1")
                GlobalContext.trace(ctx="ctx1", key="key2", value="value2")
                GlobalContext.trace(ctx="ctx1", key="key3", value="value3")

                nth_run = GlobalContext.nth_run
                assert GlobalContext.run_ctx_keys[nth_run]["ctx1"].keys() == [
                    "key1",
                    "key2",
                    "key3",
                ]

            def test_single_run_multiple_ctx_keys(self):
                GlobalContext.reset()

                GlobalContext.trace(ctx="ctx1", key="key1", value="value1")
                GlobalContext.trace(ctx="ctx1", key="key2", value="value2")
                GlobalContext.trace(ctx="ctx2", key="key3", value="value3")
                GlobalContext.trace(ctx="ctx2", key="key4", value="value4")

                nth_run = GlobalContext.nth_run
                assert GlobalContext.run_ctx_keys[nth_run]["ctx1"].keys() == [
                    "key1",
                    "key2",
                ]
                assert GlobalContext.run_ctx_keys[nth_run]["ctx2"].keys() == [
                    "key3",
                    "key4",
                ]

            def test_single_run_multiple_ctx_keys_dulicate_key(self):
                GlobalContext.reset()

                GlobalContext.trace(ctx="ctx1", key="key1", value="value1")
                GlobalContext.trace(ctx="ctx1", key="key2", value="value2")
                # Same run, different ctx, no problem
                GlobalContext.trace(ctx="ctx2", key="key2", value="value3")

                # Same run, same ctx, will print warning, and no update, problem.
                with pytest.raises(ValueError):
                    GlobalContext.trace(ctx="ctx2", key="key2", value="value4")

            def test_next_run(self):
                GlobalContext.reset()
                assert GlobalContext.nth_run == 0

                with pytest.raises(ValueError):
                    GlobalContext.next_run()
                    assert GlobalContext.nth_run == 0

                GlobalContext.trace(ctx="ctx1", key="key1", value="value1")
                GlobalContext.next_run()
                assert GlobalContext.nth_run == 1

            def test_multiple_run_and_single_trace(self):
                GlobalContext.reset()

                GlobalContext.trace(ctx="ctx1", key="key1", value="value1")
                GlobalContext.next_run()
                GlobalContext.trace(ctx="ctx1", key="key1", value="value2")

            def test_multiple_run_and_multiple_trace(self):
                GlobalContext.reset()

                GlobalContext.trace(ctx="ctx1", key="key1", value="value1")
                GlobalContext.trace(ctx="ctx1", key="key2", value="value2")
                GlobalContext.trace(ctx="ctx2", key="key3", value="value3")
                GlobalContext.next_run()
                GlobalContext.trace(ctx="ctx1", key="key1", value="value4")
                GlobalContext.trace(ctx="ctx2", key="key2", value="value5")
                GlobalContext.trace(ctx="ctx3", key="key2", value="value6")
                assert len(GlobalContext.run_ctx_keys) == 2
                assert GlobalContext.run_ctx_keys[0]["ctx1"].keys() == ["key1", "key2"]
                assert GlobalContext.run_ctx_keys[0]["ctx2"].keys() == ["key3"]
                assert GlobalContext.run_ctx_keys[1]["ctx1"].keys() == ["key1"]
                assert GlobalContext.run_ctx_keys[1]["ctx2"].keys() == ["key2"]
