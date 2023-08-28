import pytest

import io
import fastrepl

from rich.console import Console


class TestHumanEvalWithRich:
    @pytest.mark.parametrize(
        "input, expected",
        [
            ("a", "a"),
            ("a\n", "a"),
            ("c\na", "a"),
            ("c\na\n", "a"),
            ("A\nc\na", "a"),
        ],
    )
    def test_without_default(self, input, expected):
        console = Console(file=io.StringIO())

        eval = fastrepl.HumanClassifierRich(
            labels={"a": "this is a", "b": "this is b"},
            console=console,
            stream=io.StringIO(input),
        )

        actual = eval.compute("some sample")
        assert actual == expected

    @pytest.mark.parametrize(
        "input, expected",
        [
            ("a", "a"),
            ("a\n", "a"),
            ("c\n", "b"),
            ("", "b"),
        ],
    )
    def test_with_default(self, input, expected):
        console = Console(file=io.StringIO())

        eval = fastrepl.HumanClassifierRich(
            labels={"a": "this is a", "b": "this is b"},
            console=console,
            stream=io.StringIO(input),
        )

        actual = eval.compute("some sample", context="b")
        assert actual == expected
