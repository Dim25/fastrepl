import functools
from typing import List
from multiprocessing.pool import ThreadPool

from rich.progress import Progress
from datasets import Dataset

from fastrepl.eval.model.base import BaseModelEval
from fastrepl.utils import getenv

NUM_THREADS = getenv("NUM_THREADS", 8)


class Evaluator:
    __slots__ = [
        "dataset",
        "evals",
        "input_feature",
        "prediction_feature",
        "initial_context",
    ]

    def __init__(
        self,
        dataset: Dataset,
        evals: List[BaseModelEval],
        input_feature: str = "input",
        prediction_feature: str = "prediction",
    ) -> None:
        if input_feature not in dataset.features:
            raise ValueError(f"input feature {input_feature!r} not in dataset")

        self.dataset = dataset
        self.evals = evals
        self.input_feature = input_feature
        self.prediction_feature = prediction_feature

    def _run_evals(self, input: str) -> str:
        return functools.reduce(
            lambda previous, eval: eval.compute(input, previous),
            self.evals,
            self.initial_context,
        )

    def run(self, context="") -> Dataset:
        self.initial_context = context

        results = []

        with Progress() as progress:
            inputs = self.dataset[self.input_feature]
            task = progress.add_task("[cyan]Processing...", total=len(inputs))

            with ThreadPool(NUM_THREADS) as pool:
                for result in pool.imap(self._run_evals, inputs):
                    results.append(result)
                    progress.update(task, advance=1, refresh=True)

        return self.dataset.add_column(self.prediction_feature, results)
