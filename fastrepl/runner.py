from abc import ABC, abstractmethod

from rich.progress import Progress
from multiprocessing.pool import ThreadPool
from datasets import Dataset

import fastrepl
from fastrepl.utils import getenv

NUM_THREADS = getenv("NUM_THREADS", 8)


class BaseRunner(ABC):
    @abstractmethod
    def run(self) -> None:
        pass


class LocalRunner(BaseRunner):
    def __init__(
        self,
        evaluator: fastrepl.Evaluator,
        dataset: Dataset,
        input_feature: str = "input",
        output_feature: str = "prediction",
    ) -> None:
        self._evaluator = evaluator
        self._dataset = dataset
        self._input_feature = input_feature
        self._output_feature = output_feature

    def run(self) -> Dataset:
        results = []
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=len(self._dataset))

            with ThreadPool(NUM_THREADS) as pool:
                for result in pool.imap(
                    self._evaluator.run, self._dataset[self._input_feature]
                ):
                    results.append(result)
                    progress.update(task, advance=1, refresh=True)

        return self._dataset.add_column(self._output_feature, results)


class LocalRunnerREPL(LocalRunner):
    pass


class RemoteRunner(BaseRunner):
    def __init__(self) -> None:
        raise NotImplementedError


class RemoteRunnerREPL(RemoteRunner):
    pass
