from infrastructure.vectorstore.weaviate_vectorstore import WeaviateVecStore
from infrastructure.llm_clients.factory import LLMClientFactory, LLModels
from box import Box
from core.serializers import RunDetailSerializer
from core.models import Run
from services.prompts.structured_outputs import ConversationSummaryOutput, function_determinant_json_format
from services.prompts.llm_prompts import LLMPrompts, PromptType
import json
from typing import Callable, Optional
from services.grounding.linkup_retriever import LinkupGroundingRetriever
import structlog

log = structlog.get_logger(__name__)

class CoachService():
    def __init__(self, vs_name: str, user_profile: dict) -> None:
        # Core state
        self.chat_history: list[tuple[str, str]] = []
        self.session_history: list[tuple[str, str]] = []
        self.session_history_summary: Optional[str] = None

        # External resources
        self.vectorstore = WeaviateVecStore(vs_name)
        self.grounding_retriever = LinkupGroundingRetriever()
        self.llm_factory = LLMClientFactory()

        # User info
        self.user_profile = user_profile

        # Config thresholds
        self.history_summarisation_threshold = 5

    def get_raw_run_data(self,run_ids: list[int]) -> dict:
        runs = Run.objects.filter(id__in=run_ids)
        run_data = RunDetailSerializer(runs, many=True).data

        return json.dumps(run_data, indent=4)

    def get_run_summary(self, run_ids: list[int]) -> str:
        runs = Run.objects.filter(id__in=run_ids)
        run_data = RunDetailSerializer(runs, many=True).data

        system_prompt = LLMPrompts.get_prompt(PromptType.RUN_SUMMARY_GENERATOR_PROMPT, {"run_data": run_data,"user_profile": self.user_profile})
        client = self.llm_factory.get(LLModels.GEMINI_20_FLASH)

        return client.generate(
            system_prompt,
            model=LLModels.GEMINI_20_FLASH
        )

    def determine_required_functions(self, query: str) -> Box:
        chat_history = [self.session_history_summary] + self.session_history if self.session_history_summary else self.session_history
        system_prompt = LLMPrompts.get_prompt(
            PromptType.FUNCTION_DETERMINANT_PROMPT, 
            {"user_query": query,
             "user_profile": self.user_profile, 
             "chat_history": chat_history})
        
        client = self.llm_factory.get(LLModels.O4_MINI)
        output = client.generate(
            system_prompt,
            model=LLModels.O4_MINI,
            text={
            "format": function_determinant_json_format
            },
            reasoning={
                "effort": "low"
            },
            store=False
        )

        required_funcs = Box(json.loads(output))
        log.info("required_functions_resolved", required_funcs=required_funcs)

        return required_funcs


    def retrieve_necessary_context(self, query: str, status_callback: Optional[Callable[[str], None]] = None) -> dict:
        if status_callback: status_callback("Analyzing your query to determine next steps...")
        required_functions = self.determine_required_functions(query)

        context = {
            "relevant_chunks": [],
            "raw_run_data": {},
            "run_summary_data": "",
            "fact_checking_data": {},
            "query_kb_needed": required_functions.QueryKnowledgeBase_needed,
            "get_fact_check_needed": required_functions.GetGroundingAndFactCheckingData_needed
        }

        if required_functions.QueryKnowledgeBase_needed:
            if status_callback: status_callback(f"Searching knowledge base for: '{required_functions.query[:50]}...'")
            context["relevant_chunks"] = self.vectorstore.hybrid_similarity_search(required_functions.query)

        if required_functions.GetRawRunData_needed:
            if status_callback: status_callback(f"Fetching performance records for run(s): {required_functions.run_ids}...")
            context["raw_run_data"] = self.get_raw_run_data(required_functions.run_ids)

        if required_functions.GenerateRunSummary_needed:
            if status_callback: status_callback(f"Generating summary for run(s): {required_functions.run_ids}...")
            context["run_summary_data"] = self.get_run_summary(required_functions.run_ids)

        if required_functions.GetGroundingAndFactCheckingData_needed:
            # Pass the callback down
            context["fact_checking_data"] = self.grounding_retriever.retrieve_grounding_data(
                required_functions.fact_checking_query,
                status_callback=status_callback
            )

        if status_callback: status_callback("Consolidating information...")
        return context

    def close(self) -> None:
        self.vectorstore.close()
        log.info("chat_client_closed")

    def get_session_history(self) -> list:
        return self.session_history

    def summarize_session_history(self) -> None:
        conversation_messages = self.session_history[-1] if self.session_history_summary else self.session_history
        system_prompt = LLMPrompts.get_prompt(PromptType.SESSION_HISTORY_SUMMARIZATION_PROMPT, {"conversation_messages":conversation_messages})

        client = self.llm_factory.get(LLModels.GEMINI_20_FLASH)
        response = client.generate(
            system_prompt,
            model=LLModels.GEMINI_20_FLASH,
            response_mime_type="application/json",
            response_schema=ConversationSummaryOutput
        )

        conversation_summary_response: ConversationSummaryOutput = response.parsed

        if self.session_history_summary:
            self.session_history_summary = self.session_history_summary + '\n' + conversation_summary_response.conversation_summary
        else:
            self.session_history_summary = conversation_summary_response.conversation_summary

    def update_session_history(self) -> None:
        if len(self.chat_history) > self.history_summarisation_threshold:
            self.summarize_session_history()
            # Keep only the last message and the summary for context
            self.session_history = self.session_history[-1:] if self.session_history else []


    def update_history(self, question: str, answer: str) -> None:
        self.chat_history.append((f"User: {question}", f"Coach: {answer}"))
        self.session_history.append((f"User: {question}", f"Coach: {answer}"))
        self.update_session_history()

    
    def create_system_prompt(self, query: str, context: dict) -> str:
        combined_history = [self.session_history_summary] + self.session_history if self.session_history_summary else self.session_history

        if context["fact_checking_data"] == True:
             query = query + " Ground your advice and analysis using the provided `fact_checking_data` containing scientific literature search results."


        system_prompt = LLMPrompts.get_prompt(
            PromptType.COACH_PROMPT,
            {
                "query":query,
                "user_profile":self.user_profile,
                "chat_history":combined_history,
                "run_summary_data":context['run_summary_data'],
                "raw_run_data":context['raw_run_data'],
                "book_content":context['relevant_chunks'],
                "fact_checking_data": context['fact_checking_data']
            }
        )

        return system_prompt
    
    def send_question(
        self,
        query: str,
        model: LLModels,
        temperature: int | float = 1,
        **kwargs,
    ) -> str:
        prompt = self.create_system_prompt(query, self.retrieve_necessary_context(query))
        client = self.llm_factory.get(model)
        result = client.generate(
            prompt,
            model=model,
            temperature=temperature,
            **kwargs # max_tokens for claude (or max_output_tokens for openai/gemini)
        )
        self.update_history(query, result)
        return result
    
    def stream_answer(
        self,
        query: str,
        model: LLModels,
        stream_box,
        temperature: int | float = 1,
        status_callback: Optional[Callable[[str], None]] = None,
        thinking_budget: Optional[int] = None,
        is_deepthink: bool = False,
        **kwargs
    ) -> str:
        if status_callback:
            status_callback("Gathering relevant context...")

        log.info("stream_answer", is_deepthink=is_deepthink)
        
        context = self.retrieve_necessary_context(query, status_callback=status_callback)
        
        if status_callback:
            status_callback("Formulating response...")
        
        prompt = self.create_system_prompt(query, context)
        client = self.llm_factory.get(model)
        
        if status_callback:
            status_callback("Generating response...")
            
        # Configure thinking for Deepthink mode
        if is_deepthink:
            kwargs['thinking_config'] = {
                'include_thoughts': True
            }
        elif thinking_budget is not None:
            kwargs['thinking_budget'] = thinking_budget
            
        response = client.stream(
            prompt,
            model=model,
            stream_box=stream_box,
            temperature=temperature,
            **kwargs
        )
        self.update_history(query, response)
        return response