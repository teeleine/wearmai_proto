from services.segmentation.segmentation_service import SegmentationService, SegmentationOpts
from infrastructure.vectorstore.weaviate_vectorstore import WeaviateVecStore
from common.utils.text_cleaning import clean_knowledge_base
from django.core.management.base import BaseCommand
import structlog

log = structlog.get_logger(__name__)
debug = False

class Command(BaseCommand):
    help = "Vectorise the knowledge base"

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--debug", 
            action="store_true", 
            help="Enable debug mode"
        )

    def handle(self, *args, **options) -> None:
        self.debug = options.get('debug', False)
        if(self.debug):
            log.info("kb_indexing_debug_mode")

        segmentation_svc = SegmentationService()
        weaviate_vecstore = WeaviateVecStore("Bookchunks")

        # # Load book md files
        with open('wearmai/development/books/Sports Rehab Injury Prevention_clean.md') as f:
            raw = f.read()

        # # clean book
        clean_book = clean_knowledge_base(raw)

        # # save new cleaned book
        with open('wearmai/development/books/Sports Rehab Injury Prevention_clean.md', 'w') as f:
            f.write(clean_book)


        # # Chunk book content semantically and add to the vectorstore
        chunks = segmentation_svc.segment_text(
            text=clean_book,
            opt=SegmentationOpts.SDPM
        )
        
        # # add the book chunks to the vectorstore
        weaviate_vecstore.add_items(chunks)

        # test the hybrid similarity search
        log.info(weaviate_vecstore.hybrid_similarity_search("How do I test if my ankle is broken?"))
