import functools
from typing import List
from multiprocessing.pool import ThreadPool

from rich.progress import Progress
from datasets import Dataset

from fastrepl.eval.model.base import BaseModelEval
from fastrepl.utils import getenv

NUM_THREADS = getenv("NUM_THREADS", 8)


class Evaluator:
    __slots__ = ["dataset", "evals"]

    def __init__(self, dataset: Dataset, evals: List[BaseModelEval]) -> None:
        if "input" not in dataset.features:
            raise ValueError("Dataset must have input column.")

        self.dataset = dataset
        self.evals = evals

    def _run_evals(self, input: str, context="") -> str:
        return functools.reduce(
            lambda previous, eval: eval.compute(input, previous), self.evals, context
        )

    def run(self) -> Dataset:
        results = []

        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=len(self.dataset))

            with ThreadPool(NUM_THREADS) as pool:
                # TODO: we can not provide context to the first node
                for result in pool.imap(self._run_evals, self.dataset["input"]):
                    results.append(result)
                    progress.update(task, advance=1)

        return self.dataset.add_column("output", results)
