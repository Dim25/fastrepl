from typing import List, Callable, Any, TypedDict
from multiprocessing.pool import ThreadPool

from fastrepl.eval.metric.metric import BaseMetric
from fastrepl.utils import pprint


class Data(TypedDict):
    input: Any
    output: int


class Runner:
    def __init__(self, fn: Callable[..., str], worker=4) -> None:
        self.fn = fn
        self.worker = worker

    def eval(self, dataset: List[Data], metrics: List[BaseMetric]) -> dict:
        raise NotImplementedError

    def fit(self, dataset: List[Data], metrics: List[BaseMetric]) -> dict:
        raise NotImplementedError


class ClassificationRunner(Runner):
    def __init__(
        self, fn: Callable[..., str], label_mapping: dict[str, int], worker=4
    ) -> None:
        super().__init__(fn, worker)
        self.label_mapping = label_mapping

    def eval(self, dataset: List[Data], metrics: List[BaseMetric]) -> dict:
        prediction: List[int] = []
        reference: List[int] = []
        report: dict[str, float] = {}

        with ThreadPool(processes=self.worker) as pool:
            try:
                async_results = [pool.apply_async(self.fn, d["input"]) for d in dataset]

                for i, result in enumerate(async_results):
                    label_text = result.get()
                    label_int = self.label_mapping.get(label_text)
                    if label_int is None:
                        pprint(
                            f"[yellow]warning[/yellow]: {label_text} not in {self.label_mapping}, skipping {i}th data"
                        )
                    else:
                        prediction.append(label_int)
                        reference.append(dataset[i]["output"])

            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                return {}

        for metric in metrics:
            report[metric.name] = metric.compute(reference, prediction)

        return report

    def fit(self, dataset: List[Data], metrics: List[BaseMetric]) -> dict:
        return {}
