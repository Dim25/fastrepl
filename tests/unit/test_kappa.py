import pytest

from sklearn.metrics import confusion_matrix
from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa, aggregate_raters

from fastrepl.utils import kappa


class TestCohensKappa:
    def test_basic1(self):
        result = cohens_kappa(
            table=[[1, 2, 3], [2, 3, 3], [1, 1, 2]], return_results=False
        )
        assert result == pytest.approx(0.01818, abs=1e-5)

    def test_basic2(self):
        result = cohens_kappa(
            table=[[1, 2, 3], [1, 2, 3], [1, 2, 3]], return_results=False
        )
        assert result == pytest.approx(0.0)


class TestFleissKappa:
    def test_aggregate_raters(self):
        table, categories = aggregate_raters(
            [[0, 1, 2], [1, 0, 1], [2, 2, 0], [1, 0, 2]]
        )

        assert (table == [[1, 1, 1], [1, 2, 0], [1, 0, 2], [1, 1, 1]]).all()
        assert (categories == [0, 1, 2]).all()


@pytest.mark.parametrize(
    "predictions, result",
    [
        ([[1, None], [1, 2]], 0.333),
        ([[1, None], [1, None]], 1),
        ([[1, 2], [1, 2]], 1.0),
        ([[1, 2, 3], [3, 2, 3]], 0.499),
        ([[1, 2, 3], [3, 2, 1]], 0),
        ([[1, 2, 3, 4], [4, 3, 2, 1]], -0.333),
        ([["POSITIVE", "NEGATIVE"], ["NEGATIVE", "POSITIVE"]], -1.0),
        ([["POSITIVE", "NEGATIVE"], ["POSITIVE", "NEGATIVE"]], 1.0),
        (
            [
                ["POSITIVE", "NEGATIVE", "NEGATIVE"],
                ["POSITIVE", "POSITIVE", "NEGATIVE"],
            ],
            0.399,
        ),
        (
            [
                ["A", "B", "C"],
                ["A", "B", "B"],
            ],
            0.499,
        ),
    ],
)
def test_kappa(predictions, result):
    assert kappa(*predictions) == pytest.approx(result, abs=1e-3)
