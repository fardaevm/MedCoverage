from abc import ABC, abstractmethod
from typing import List

class BaseRetriever(ABC):
    @abstractmethod
    def search(query: str, top_k: int = 5) -> List[str]:
        pass


