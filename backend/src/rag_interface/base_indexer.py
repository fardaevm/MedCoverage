from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from .base_datastore import DataItem


class BaseIndexer(ABC):
    @abstractmethod
    def index(self, document_path: List[str]) -> List[DataItem]:
        pass