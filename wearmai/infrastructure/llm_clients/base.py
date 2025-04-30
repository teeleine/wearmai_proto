from abc import ABC, abstractmethod
from typing import Iterator
from enum import StrEnum

class LLModels(StrEnum):
    GEMINI_25_FLASH = "gemini-2.5-flash-preview-04-17"
    GEMINI_20_FLASH = "gemini-2.0-flash"
    GPT_41 = "gpt-4.1"
    CLAUDE_37_SONNET = "claude-3-7-sonnet-20250219"
    O4_MINI = "o4-mini"


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a full response."""
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream response chunks."""
        pass