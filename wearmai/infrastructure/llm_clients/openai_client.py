from .base import BaseLLMClient
from streamlit.delta_generator import DeltaGenerator
from openai import OpenAI
from time import sleep

class OpenAIClient(BaseLLMClient):
    def __init__(self):
        self.client = OpenAI()

    def generate(self, prompt: str, model: str, **kwargs) -> str:
        response = self.client.responses.create(
            model=model,
            input=[{"role": "developer", "content": [
                    {
                    "type": "input_text",
                    "text": prompt
                    }
                ]}],
            **kwargs
        )
        return response.output_text

    def stream(self, prompt: str, model: str, stream_box: DeltaGenerator, **kwargs) -> str:
        final_response = ""
        stream = self.client.responses.create(
            model=model,
            input=[{"role": "developer", "content": prompt}],
            stream=True,
            **kwargs
        )
        for event in stream:
            if event.type == "response.output_text.delta" and event.delta:
                final_response += event.delta
                sleep(0.002)
                stream_box.markdown(final_response + "▌")
        stream_box.markdown(final_response)
        return final_response