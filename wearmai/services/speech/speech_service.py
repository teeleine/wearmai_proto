import os
import tempfile
from openai import OpenAI
from wearmai.settings import OPENAI_API_KEY

class SpeechService:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
    def transcribe_audio(self, audio_data: bytes, file_extension: str = "wav") -> str:
        """
        Transcribe audio data using OpenAI's Whisper model.
        
        Args:
            audio_data: Raw audio data in bytes
            file_extension: The file extension/format of the audio (e.g., "wav", "webm", "mp3")
            
        Returns:
            str: The transcribed text
        """
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=True) as temp_file:
            # Write the audio data to the temporary file
            temp_file.write(audio_data)
            temp_file.flush()
            
            # Open the file in binary read mode for the API
            with open(temp_file.name, "rb") as audio_file:
                try:
                    # Call OpenAI's transcription API
                    transcription = self.client.audio.transcriptions.create(
                        model="whisper-1",  # Using Whisper model
                        file=audio_file,
                        response_format="text"
                    )
                    return transcription
                except Exception as e:
                    print(f"Error transcribing audio: {str(e)}")
                    return "" 