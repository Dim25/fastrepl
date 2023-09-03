from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod


class BaseEvalWithReference(ABC):
    @abstractmethod
    def compute(
        self, predictions: List[Any], references: List[Any], **kwargs
    ) -> Dict[str, Any]:
        ...


class BaseEvalWithoutReference(ABC):
    @abstractmethod
    def compute(self, prediction: str, context: Optional[str]) -> Optional[str]:
        ...
