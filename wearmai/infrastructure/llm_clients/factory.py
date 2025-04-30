from .base import BaseLLMClient, LLModels
from typing import Dict
from .gemini_client import GeminiClient
from .claude_client import ClaudeClient
from .openai_client import OpenAIClient
import os

class LLMClientFactory:
    _registry: Dict[LLModels, BaseLLMClient] = {}

    @classmethod
    def register(cls, model: LLModels, client: BaseLLMClient):
        cls._registry[model] = client

    @classmethod
    def get(cls, model: LLModels) -> BaseLLMClient:
        return cls._registry[model]
    
LLMClientFactory.register(LLModels.GEMINI_20_FLASH,GeminiClient(os.getenv("GEMINI_API_KEY")))
LLMClientFactory.register(LLModels.GEMINI_25_FLASH,GeminiClient(os.getenv("GEMINI_API_KEY")))
LLMClientFactory.register(LLModels.CLAUDE_37_SONNET, ClaudeClient(os.getenv("ANTHROPIC_API_KEY")))
LLMClientFactory.register(LLModels.O4_MINI, OpenAIClient())
LLMClientFactory.register(LLModels.GPT_41, OpenAIClient())