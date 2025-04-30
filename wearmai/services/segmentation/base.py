from enum import StrEnum
from abc import ABC, abstractmethod
from typing import List, Any

class SegmentationOpts(StrEnum):
    SDPM = "SDPMChunker"
    SEMANTIC = "SemanticChunker"
    LATE = "LateChunker"


class BaseSegmenter(ABC):
    @abstractmethod
    def chunk(self, text: str, **kwargs: Any) -> List[str]:
        """Return list of text chunks"""