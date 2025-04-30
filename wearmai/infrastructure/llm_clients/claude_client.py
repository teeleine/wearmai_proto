from .base import BaseLLMClient
from streamlit.delta_generator import DeltaGenerator
import anthropic
from time import sleep

class ClaudeClient(BaseLLMClient):
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, model: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content

    def stream(self, prompt: str, model: str, stream_box: DeltaGenerator, **kwargs) -> str:
        final_response = ""
        with self.client.messages.stream(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        ) as stream:
            for text in stream.text_stream:
                final_response += text
                sleep(0.002)
                stream_box.markdown(final_response + "▌")
        stream_box.markdown(final_response)
        return final_response