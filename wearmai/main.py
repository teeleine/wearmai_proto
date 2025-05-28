import streamlit as st
import coach_ui  # This will run your Streamlit app
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.speech_endpoints import app as speech_app

app = FastAPI()

# Mount the speech API
app.mount("/api/speech", speech_app)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and mount your Streamlit app
def run_streamlit():
    import coach_ui  # This will run your Streamlit app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 