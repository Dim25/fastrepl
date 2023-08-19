from fastrepl.eval.model import ScoringModelEval


class TestScoringModelEval:
    def test_abcde(self):
        result = ScoringModelEval().compute(
            "C",
            [
                ("A", "1"),
                ("B", "2"),
            ],
        )
        assert result == "3"
