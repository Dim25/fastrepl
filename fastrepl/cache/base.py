from abc import ABC, abstractmethod
from typing import Optional


class BaseCache(ABC):
    @abstractmethod
    def lookup(self, model: str, prompt: str) -> Optional[str]:
        pass

    @abstractmethod
    def update(self, model: str, prompt: str, response: str) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
