from .base import BaseSegmenter, SegmentationOpts
from typing import Dict

class SegmenterFactory:
    _registry: Dict[SegmentationOpts, BaseSegmenter] = {}

    @classmethod
    def register(cls, opt: SegmentationOpts, segmenter: BaseSegmenter) -> None:
        cls._registry[opt] = segmenter

    @classmethod
    def get(cls, opt: SegmentationOpts) -> BaseSegmenter:
        if opt not in cls._registry:
            raise ValueError(f"No segmenter registered for {opt}")
        return cls._registry[opt]