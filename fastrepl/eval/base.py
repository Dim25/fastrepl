from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod


class BaseMetaEvalNode(ABC):
    @abstractmethod
    def compute(
        self, predictions: List[Any], references: List[Any], **kwargs
    ) -> Dict[str, Any]:
        ...


class BaseEvalNode(ABC):
    @abstractmethod
    def compute(self, prediction: str, context: Optional[str]) -> Optional[str]:
        ...
