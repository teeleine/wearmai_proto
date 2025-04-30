# services/llm_clients/gemini_client.py
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from .base import BaseLLMClient, LLModels
from time import sleep
import structlog

log = structlog.get_logger(__name__)

class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str,
        model: LLModels | str,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        thinking_budget: int | None = None,
        response_mime_type: str | None = None,
        response_schema: type | None = None,
    ) -> str | object:
        """
        Non-streaming call to Gemini. Wraps all optional parameters into a single config.
        Returns response.parsed if response_schema provided, else response.text.
        """
        # Build config kwargs
        config_kwargs: dict = {}
        if max_output_tokens is not None:
            config_kwargs['max_output_tokens'] = max_output_tokens
        if temperature is not None:
            config_kwargs['temperature'] = temperature
        if top_p is not None:
            config_kwargs['top_p'] = top_p
        if response_mime_type is not None:
            config_kwargs['response_mime_type'] = response_mime_type
        if response_schema is not None:
            config_kwargs['response_schema'] = response_schema

        # Instantiate the SDK config
        config = GenerateContentConfig(**config_kwargs)

        # Perform the call
        response = self.client.models.generate_content(
            model=(model.value if isinstance(model, LLModels) else model),
            contents=prompt,
            config=config,
        )

        # Return parsed vs raw text
        if response_schema is not None:
            return response.parsed
        return response.text

    def stream(
        self,
        prompt: str,
        model: LLModels | str,
        stream_box,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        thinking_budget: int | None = None
    ) -> str:
        """
        Streaming call to Gemini. Writes chunks to stream_box.markdown.
        Returns the full assembled response string.
        """
        # Build config kwargs
        config_kwargs: dict = {}
        if max_output_tokens is not None:
            config_kwargs['max_output_tokens'] = max_output_tokens
        if temperature is not None:
            config_kwargs['temperature'] = temperature
        if top_p is not None:
            config_kwargs['top_p'] = top_p
        if thinking_budget is not None:
            config_kwargs['thinking_config'] = ThinkingConfig(thinking_budget=thinking_budget)
            log.info("thinking_budget_used", budget=thinking_budget)

        config = GenerateContentConfig(**config_kwargs)

        final_response = ''
        stream = self.client.models.generate_content_stream(
            model=(model.value if isinstance(model, LLModels) else model),
            contents=prompt,
            config=config,
        )

        for chunk in stream:
            if chunk.text:
                final_response += chunk.text
                # small sleep for smoother cursor effect
                sleep(0.002)
                stream_box.markdown(final_response + '▌')
        # final render without cursor
        stream_box.markdown(final_response)
        return final_response
