from typing import Any, List, Optional
from abc import ABC, abstractmethod


class BaseEvalWithReference(ABC):
    @abstractmethod
    def compute(self, predictions: List[Any], references: List[Any], **kwargs):
        ...


class BaseEvalWithoutReference(ABC):
    @abstractmethod
    def compute(self, prediction: str, context: str) -> Optional[str]:
        ...

    @abstractmethod
    def is_interactive(self) -> bool:
        ...
