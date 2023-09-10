import pytest
import warnings

import fastrepl
from datasets import Dataset


@pytest.fixture
def mock_runs(monkeypatch):
    def ret(values):
        iter_values = iter(values)

        def mock_run(*args, **kwargs):
            return next(iter_values)

        monkeypatch.setattr(fastrepl.LocalRunner, "_run", mock_run)

    return ret


def test_runner_num_1(mock_runs):
    mock_runs([[1]])

    ds = fastrepl.LocalRunner(
        evaluator=[fastrepl.LLMClassificationHead(context="", labels={})],
        dataset=Dataset.from_dict({"input": [1]}),
    ).run(num=1)

    assert ds.column_names == ["input", "prediction"]


def test_runner_num_2_without_warning(mock_runs):
    mock_runs([[1, 2, 3, 4], [1, 2, 3, 5]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        ds = fastrepl.LocalRunner(
            evaluator=[fastrepl.LLMClassificationHead(context="", labels={})],
            dataset=Dataset.from_dict({"input": [1, 2, 3, 4]}),
        ).run(num=2)

    assert ds.column_names == ["input", "prediction"]
    assert ds["prediction"] == [[1, 1], [2, 2], [3, 3], [4, 5]]


def test_runner_num_2_with_warning(mock_runs):
    mock_runs([[4, 3, 2, 1], [2, 1, 3, 5]])

    with pytest.warns():
        ds = fastrepl.LocalRunner(
            evaluator=[fastrepl.LLMClassificationHead(context="", labels={})],
            dataset=Dataset.from_dict({"input": [1, 2, 3, 4]}),
        ).run(num=2)

    assert ds.column_names == ["input", "prediction"]
    assert ds["prediction"] == [[4, 2], [3, 1], [2, 3], [1, 5]]


def test_runner_num_2_handle_none(mock_runs):
    mock_runs([[1, 2, 3, 4], [1, 2, 3, None]])

    ds = fastrepl.LocalRunner(
        evaluator=[fastrepl.LLMClassificationHead(context="", labels={})],
        dataset=Dataset.from_dict({"input": [1, 2, 3, 4]}),
    ).run(num=2)

    assert ds.column_names == ["input", "prediction"]
    assert ds["prediction"] == [[1, 1], [2, 2], [3, 3], [4, None]]
