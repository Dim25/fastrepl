from typing import Optional, Any, List
from abc import ABC, abstractmethod


class BaseEvalWithReference(ABC):
    @abstractmethod
    def compute(self, predictions: List[Any], references: List[Any], **kwargs):
        ...


class BaseEvalWithoutReference(ABC):
    @abstractmethod
    def compute(self, prediction: str, context: Optional[str]) -> Optional[str]:
        ...
