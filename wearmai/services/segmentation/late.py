from typing import Any, List
from chonkie import LateChunker
from .base import BaseSegmenter

class LateSegmenter(BaseSegmenter):
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1024,
        min_characters_per_chunk: int = 24,
    ):
        self.chunker = LateChunker(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            min_characters_per_chunk=min_characters_per_chunk,
        )

    def chunk(self, text: str, **kwargs: Any) -> List[str]:
        if kwargs:
            for key, val in kwargs.items():
                setattr(self.chunker, key, val)
        return self.chunker.chunk(text)