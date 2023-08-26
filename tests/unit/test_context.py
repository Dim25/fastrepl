import pytest

from fastrepl.loop import REPLContext, AnalyzeContext


class TestAnalyzeContext:
    def test_empty(self):
        AnalyzeContext.reset()
        assert AnalyzeContext.nth_run == 0
        assert len(AnalyzeContext.run_ctx_keys) == 0

    def test_single_run_ctx(self):
        AnalyzeContext.reset()
        AnalyzeContext.trace(ctx="ctx1", key="key1", value="value1")
        assert len(AnalyzeContext.run_ctx_keys) == 1

        nth_run = AnalyzeContext.nth_run
        assert AnalyzeContext.run_ctx_keys[nth_run]["ctx1"].keys() == ["key1"]

    def test_single_run_ctx_multiple_keys(self):
        AnalyzeContext.reset()

        AnalyzeContext.trace(ctx="ctx1", key="key1", value="value1")
        AnalyzeContext.trace(ctx="ctx1", key="key2", value="value2")
        AnalyzeContext.trace(ctx="ctx1", key="key3", value="value3")

        nth_run = AnalyzeContext.nth_run
        assert AnalyzeContext.run_ctx_keys[nth_run]["ctx1"].keys() == [
            "key1",
            "key2",
            "key3",
        ]

    def test_single_run_multiple_ctx_keys(self):
        AnalyzeContext.reset()

        AnalyzeContext.trace(ctx="ctx1", key="key1", value="value1")
        AnalyzeContext.trace(ctx="ctx1", key="key2", value="value2")
        AnalyzeContext.trace(ctx="ctx2", key="key3", value="value3")
        AnalyzeContext.trace(ctx="ctx2", key="key4", value="value4")

        nth_run = AnalyzeContext.nth_run
        assert AnalyzeContext.run_ctx_keys[nth_run]["ctx1"].keys() == [
            "key1",
            "key2",
        ]
        assert AnalyzeContext.run_ctx_keys[nth_run]["ctx2"].keys() == [
            "key3",
            "key4",
        ]

    def test_single_run_multiple_ctx_keys_dulicate_key(self):
        AnalyzeContext.reset()

        AnalyzeContext.trace(ctx="ctx1", key="key1", value="value1")
        AnalyzeContext.trace(ctx="ctx1", key="key2", value="value2")
        # Same run, different ctx, same key = no problem
        AnalyzeContext.trace(ctx="ctx2", key="key2", value="value3")

        # Same run, ctx, and key = warning
        with pytest.warns(UserWarning):  # TODO: this cause ResourceWarning
            AnalyzeContext.trace(ctx="ctx2", key="key2", value="value4")

    def test_next_run(self):
        AnalyzeContext.reset()
        assert AnalyzeContext.nth_run == 0

        with pytest.warns(UserWarning):  # TODO: this cause ResourceWarning
            AnalyzeContext.next_run()
        assert AnalyzeContext.nth_run == 1

    def test_multiple_run_and_single_trace(self):
        AnalyzeContext.reset()

        AnalyzeContext.trace(ctx="ctx1", key="key1", value="value1")
        assert AnalyzeContext.nth_run == 0
        assert len(AnalyzeContext.run_ctx_keys) == 1
        assert AnalyzeContext.run_ctx_keys[0]["ctx1"].keys() == ["key1"]

        AnalyzeContext.next_run()

        AnalyzeContext.trace(ctx="ctx1", key="key1", value="value2")
        assert AnalyzeContext.nth_run == 1
        assert len(AnalyzeContext.run_ctx_keys) == 2
        assert AnalyzeContext.run_ctx_keys[0]["ctx1"].keys() == ["key1"]

    def test_multiple_run_and_multiple_trace(self):
        AnalyzeContext.reset()

        AnalyzeContext.trace(ctx="ctx1", key="key1", value="value1")
        AnalyzeContext.trace(ctx="ctx1", key="key2", value="value2")
        AnalyzeContext.trace(ctx="ctx2", key="key3", value="value3")
        AnalyzeContext.next_run()
        AnalyzeContext.trace(ctx="ctx1", key="key1", value="value4")
        AnalyzeContext.trace(ctx="ctx2", key="key2", value="value5")
        AnalyzeContext.trace(ctx="ctx3", key="key2", value="value6")
        assert len(AnalyzeContext.run_ctx_keys) == 2
        assert AnalyzeContext.run_ctx_keys[0]["ctx1"].keys() == ["key1", "key2"]
        assert AnalyzeContext.run_ctx_keys[0]["ctx2"].keys() == ["key3"]
        assert AnalyzeContext.run_ctx_keys[1]["ctx1"].keys() == ["key1"]
        assert AnalyzeContext.run_ctx_keys[1]["ctx2"].keys() == ["key2"]


class TestREPLContext:
    def test_empty(self):
        REPLContext.reset()
        assert REPLContext.head_id == REPLContext.latest_id
        assert len(REPLContext.ctx_key_values) == 0

    def test_single_trace(self):
        REPLContext.reset()
        REPLContext.trace(ctx="ctx1", key="key1", value="value1")
        assert len(REPLContext.ctx_key_values) == 1
        assert REPLContext.get_current_value(ctx="ctx1", key="key1") == "value1"

    def test_multiple_trace(self):
        REPLContext.reset()
        REPLContext.trace(ctx="ctx1", key="key1", value="value1")
        REPLContext.trace(ctx="ctx1", key="key1", value="value1")
