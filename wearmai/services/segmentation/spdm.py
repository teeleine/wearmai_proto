from typing import Any, List
from chonkie import SDPMChunker
from .base import BaseSegmenter

class SDPMSegmenter(BaseSegmenter):
    def __init__(
        self,
        embedding_model: str = "minishlab/potion-base-32M",
        threshold: float = 0.5,
        chunk_size: int = 1024,
        min_sentences: int = 1,
        skip_window: int = 1,
    ):
        self.chunker = SDPMChunker(
            embedding_model=embedding_model,
            threshold=threshold,
            chunk_size=chunk_size,
            min_sentences=min_sentences,
            skip_window=skip_window,
        )

    def chunk(self, text: str, **kwargs: Any) -> List[str]:
        # kwargs can override defaults
        if kwargs:
            for key, val in kwargs.items():
                setattr(self.chunker, key, val)
        return self.chunker.chunk(text)