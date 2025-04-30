from linkup import LinkupClient
from .base import BaseGroundingRetriever
import os
from typing import Optional, Callable
from services.prompts.llm_prompts import LLMPrompts, PromptType
import structlog

log = structlog.get_logger(__name__)


class LinkupGroundingRetriever(BaseGroundingRetriever):
    def __init__(
        self,
        depth: str = "standard",
        output_type: str = "searchResults"
    ):
        self.linkup_client = LinkupClient(api_key=os.getenv("LINKUP_API_KEY"))
        self.depth = depth
        self.output_type = output_type
    
    def retrieve_grounding_data(
        self, 
        search_query: str, 
        status_callback: Optional[Callable[[str], None]] = None
    ) -> dict:
        if status_callback: status_callback(f"Fact-checking output with academic sources using the search term: '{search_query[:50]}...'")

        final_search_query = LLMPrompts.get_prompt(PromptType.FACT_CHECKING_SEARCH_QUERY_PROMPT, {"search_query":search_query})

        try:
            search_response = self.linkup_client.search(
                query=final_search_query,
                depth=self.depth,
                output_type=self.output_type # can be sourcedAnswer (llm-generated answer based on sources) or searchResults, which is faster as it's just the raw search results
            )
            log.info("linkup_search_response", search_response=search_response)
            return search_response
        
        except Exception as e:
            log.exception(f"Error during Linkup Search", error=e)
            
            if status_callback: status_callback(f"Fact-checking output with online academic sources failed.")
            return {"error": str(e), "answer": "Could not fact-check output with online academic sources."}
        