from abc import ABC, abstractmethod
from typing import Optional, List

from multiprocessing.pool import ThreadPool
from datasets import Dataset
from rich.progress import Progress, TaskID

import fastrepl
from fastrepl.utils import getenv, kappa
from fastrepl.warnings import warn, InconsistentPredictionWarning

NUM_THREADS = getenv("NUM_THREADS", 8)


class BaseRunner(ABC):
    @abstractmethod
    def run(self) -> Dataset:
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

    def _run_eval(self, sample: str) -> Optional[str]:
        return self._evaluator.run(sample)

    def _run(self, progress: Progress, task_id: TaskID) -> List[Optional[str]]:
        results = []
        with ThreadPool(NUM_THREADS) as pool:
            for result in pool.imap(self._run_eval, self._dataset[self._input_feature]):
                results.append(result)
                progress.update(task_id, advance=1, refresh=True)
        return results

    def run(self, num=1) -> Dataset:
        with Progress() as progress:
            task_id = progress.add_task(
                "[cyan]Processing...",
                total=len(self._dataset) * num,
            )

            if num == 1:
                return self._dataset.add_column(
                    self._output_feature,
                    self._run(progress, task_id),
                )
            elif num == 2:
                predictions = [self._run(progress, task_id) for _ in range(num)]

                value = kappa(*predictions)
                if value < 0.4:
                    warn(InconsistentPredictionWarning, context=str(value))

                return self._dataset.add_column(
                    self._output_feature, list(zip(*predictions))
                )
            else:
                raise NotImplementedError


class LocalRunnerREPL(LocalRunner):
    pass


class RemoteRunner(BaseRunner):
    def __init__(self) -> None:
        raise NotImplementedError
