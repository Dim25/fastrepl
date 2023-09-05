import pytest
import warnings
import litellm.gpt_cache

import fastrepl


@pytest.fixture
def mock_completion(monkeypatch):
    def ret(values):
        iter_values = iter(values)

        def mock(*args, **kwargs):
            return {
                "choices": [
                    {"message": {"content": next(iter_values)}, "finish_reason": "stop"}
                ]
            }

        monkeypatch.setattr(litellm.gpt_cache, "completion", mock)

    return ret


class TestLLMClassificationHeadBasic:
    def test_return_result(self, mock_completion):
        eval = fastrepl.LLMClassificationHead(
            context="test",
            labels={
                "POSITIVE": "this is positive",
                "NEGATIVE": "this is negative",
            },
        )

        mock_completion([eval.mapping[0].token])
        assert eval.compute("") == eval.mapping[0].label

    def test_return_none(self, mock_completion):
        eval = fastrepl.LLMClassificationHead(
            context="test",
            labels={
                "POSITIVE": "this is positive",
                "NEGATIVE": "this is negative",
            },
        )

        mock_completion(["D"])

        with pytest.warns():
            assert eval.compute("") is None


class TestClassificationHeadShuffle:
    def test_basic(self):
        pass


class TestClassificationHeadConsensus:
    def test_no_need_for_consensus(self, mock_completion):
        eval = fastrepl.LLMClassificationHead(
            context="test",
            labels={"POSITIVE": "this is positive", "NEGATIVE": "this is negative"},
            position_debias_strategy="consensus",
        )

        mock_completion([eval.mapping[-1].token])
        assert eval.compute("") is not None

    def test_triggered_twice(self, mock_completion):
        eval = fastrepl.LLMClassificationHead(
            context="test",
            labels={"POSITIVE": "this is positive", "NEGATIVE": "this is negative"},
            position_debias_strategy="consensus",
        )

        mock_completion([eval.mapping[0].token])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StopIteration):
                eval.compute("")

    def test_consensus_success(self, mock_completion):
        eval = fastrepl.LLMClassificationHead(
            context="test",
            labels={"POSITIVE": "this is positive", "NEGATIVE": "this is negative"},
            position_debias_strategy="consensus",
        )

        mock_completion([eval.mapping[0].token, eval.mapping[0].token])
        assert eval.compute("") is not None

    def test_consensus_failed(self, mock_completion):
        eval = fastrepl.LLMClassificationHead(
            context="test",
            labels={"POSITIVE": "this is positive", "NEGATIVE": "this is negative"},
            position_debias_strategy="consensus",
        )

        mock_completion([eval.mapping[0].token, eval.mapping[1].token])
        assert eval.compute("") is None


class TestLLMGradingHead:
    @pytest.mark.parametrize(
        "return_value, number_from, number_to",
        [
            ("2", 1, 3),
            ("3", 1, 5),
        ],
    )
    def test_return_result(self, mock_completion, return_value, number_from, number_to):
        eval = fastrepl.LLMGradingHead(
            context="test",
            number_from=number_from,
            number_to=number_to,
        )

        mock_completion([return_value])
        assert eval.compute("") == return_value

    @pytest.mark.parametrize(
        "return_value, number_from, number_to",
        [
            ("1.1", 1, 3),
            ("2.2", 1, 3),
        ],
    )
    def test_return_result_with_warnings(
        self, mock_completion, return_value, number_from, number_to
    ):
        eval = fastrepl.LLMGradingHead(
            context="test",
            number_from=number_from,
            number_to=number_to,
        )

        mock_completion([return_value])

        with pytest.warns():
            assert eval.compute("") == return_value

    @pytest.mark.parametrize(
        "return_value, number_from, number_to",
        [
            ("-1", 1, 3),
            ("0", 1, 3),
            ("6", 1, 5),
            ("4.1", 1, 3),
        ],
    )
    def test_return_none(self, mock_completion, return_value, number_from, number_to):
        eval = fastrepl.LLMGradingHead(
            context="test",
            number_from=number_from,
            number_to=number_to,
        )

        mock_completion([return_value])

        with pytest.warns():
            assert eval.compute("") is None
