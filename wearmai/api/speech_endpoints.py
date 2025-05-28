from fastapi import FastAPI, UploadFile, File
from services.speech.speech_service import SpeechService

app = FastAPI()
speech_service = SpeechService()

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Endpoint to handle audio transcription requests.
    """
    try:
        # Read the audio file content
        audio_content = await audio.read()
        
        # Get the file extension from the filename
        file_extension = audio.filename.split('.')[-1]
        
        # Transcribe the audio
        transcription = speech_service.transcribe_audio(audio_content, file_extension)
        
        return transcription
    except Exception as e:
        return {"error": str(e)} 