from .factory import SegmenterFactory
from .base import SegmentationOpts
from .spdm import SDPMSegmenter
from .semantic import SemanticSegmenter
from .late import LateSegmenter
from typing import Any, List

class SegmentationService:
    def __init__(self, factory: SegmenterFactory = SegmenterFactory) -> None:
        self._factory = factory

    def segment_text(self, text: str, opt: SegmentationOpts, **kwargs: Any) -> List[str]:
        if not isinstance(opt, SegmentationOpts):
            raise ValueError(f"Invalid segmentation option: {opt}")
        segmenter = self._factory.get(opt)
        return segmenter.chunk(text, **kwargs)
    

SegmenterFactory.register(SegmentationOpts.SDPM, SDPMSegmenter())
SegmenterFactory.register(SegmentationOpts.SEMANTIC, SemanticSegmenter())
SegmenterFactory.register(SegmentationOpts.LATE, LateSegmenter())