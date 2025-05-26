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
        thinking_budget: int | None = None,
        thinking_config: dict | None = None
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
        elif thinking_config is not None:
            config_kwargs['thinking_config'] = ThinkingConfig(**thinking_config)
            log.info("thinking_config_used", config=thinking_config)

        config = GenerateContentConfig(**config_kwargs)

        final_response = ''
        stream = self.client.models.generate_content_stream(
            model=(model.value if isinstance(model, LLModels) else model),
            contents=prompt,
            config=config,
        )

        for chunk in stream:
            # Handle thinking output if available
            if hasattr(chunk, 'candidates') and chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if not part.text:
                        continue
                    elif hasattr(part, 'thought') and part.thought:
                        # Don't add thoughts to final response, but update stream box
                        stream_box.markdown(final_response + "\n\n💭 " + part.text + "\n\n▌")
                    else:
                        final_response += part.text
                        stream_box.markdown(final_response + "▌")
            # Regular text output
            elif hasattr(chunk, 'text') and chunk.text:
                final_response += chunk.text
                stream_box.markdown(final_response + "▌")
                
        # final render without cursor
        stream_box.markdown(final_response)
        return final_response
