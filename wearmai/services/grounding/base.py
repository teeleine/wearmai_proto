from abc import ABC, abstractmethod
from typing import Optional, Callable

class BaseGroundingRetriever(ABC):

    @abstractmethod
    def retrieve_grounding_data(self, search_query: str, status_callback: Optional[Callable[[str], None]] = None, **kwargs) -> dict:
        """Retrieve grounding data"""
        pass
