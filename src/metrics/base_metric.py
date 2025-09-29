from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    def __init__(self, name: str | None = None, **kwargs: Any):
        self.name = name or type(self).__name__

    @abstractmethod
    def __call__(self, **batch) -> float:
        raise NotImplementedError
