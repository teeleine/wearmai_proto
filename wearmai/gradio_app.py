import os
import django
from infrastructure.logging import configure_logging
import plotly.express as px
import plotly.io
from services.speech.speech_service import SpeechService
import gradio as gr
import time
import json

# Configure Django settings before importing any Django models
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "wearmai.settings")
django.setup()

configure_logging()

from services.llm_coach.coach_service import CoachService
from user_profile.loader import load_profile
from infrastructure.llm_clients.base import LLModels
from django.db import connection
import structlog

log = structlog.get_logger(__name__)

# Initialize global services
speech_service = SpeechService()
user_profile_data = None
coach_svc = None

class StreamCollector:
    """Mock stream_box object that collects streamed text"""
    def __init__(self):
        self.content = ""
        self.chunks = []
    
    def markdown(self, text):
        """Mimic Streamlit's markdown method"""
        # Remove the cursor character if present
        clean_text = text.replace("▌", "")
        self.content = clean_text
        self.chunks.append(clean_text)
    
    def get_content(self):
        return self.content
    
    def get_chunks(self):
        return self.chunks

def initialize_services():
    """Initialize services on startup"""
    global user_profile_data, coach_svc
    
    try:
        if not connection.is_usable():
            connection.close()
            connection.connect()
        user_profile_data = load_profile(name="Test User 2 - Full Data Load")
        
        if user_profile_data and "llm_user_profile" in user_profile_data:
            coach_svc = CoachService(
                vs_name="Bookchunks",
                user_profile=user_profile_data["llm_user_profile"],
            )
        else:
            log.error("llm_user_profile missing", data=user_profile_data)
    except Exception as e:
        log.error("Error loading user profile", exc_info=e)
        user_profile_data = None

# Initialize services
initialize_services()

# Quick actions
quick_actions = {
    "📊 Analyze my last run": "Can you analyze my last run data and provide detailed insights about my performance?",
    "🎯 Create a 4-week plan": "Can you create a personalized 4-week running plan tailored to my current fitness level?",
    "🏃‍♂️ Analyze run technique": "Can you analyze my running technique and provide specific recommendations for improvement?",
    "🚨 Injury risk assessment": "Can you perform a comprehensive injury risk assessment based on my running data and patterns?",
}

# Store plots globally for access
plot_storage = {}

def add_message(history, message, mode):
    """Add user message to chat history"""
    if not history:
        history = []
    
    # Handle audio files
    for file in message.get("files", []):
        if file.endswith(('.wav', '.mp3', '.m4a', '.webm')):
            # Transcribe audio
            try:
                with open(file, 'rb') as f:
                    audio_data = f.read()
                transcription = speech_service.transcribe_audio(audio_data, "wav")
                if transcription:
                    history.append({"role": "user", "content": transcription})
            except Exception as e:
                log.error("Error transcribing audio", exc_info=e)
                history.append({"role": "user", "content": f"[Error transcribing audio: {e}]"})
        else:
            # For other files, just note them
            history.append({"role": "user", "content": f"[File uploaded: {file}]"})
    
    # Add text message if present
    if message.get("text"):
        history.append({"role": "user", "content": message["text"]})
    
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def process_bot_response(history, mode):
    """Process the bot response with streaming"""
    if not history or not coach_svc:
        error_msg = "System not ready. Please try again later."
        history.append({"role": "assistant", "content": error_msg})
        yield history
        return
    
    # Get the last user message
    user_messages = [msg for msg in history if msg["role"] == "user"]
    if not user_messages:
        return history
    
    last_user_msg = user_messages[-1]["content"]
    
    # Initialize assistant message (we'll build it up)
    assistant_messages = []
    current_text_content = ""
    
    # Variables for tracking thinking content
    thinking_content = ""
    thoughts_seen = set()
    has_plot = False
    stream_collector = StreamCollector()
    plot_fig = None
    
    # Add initial empty text message
    history.append({"role": "assistant", "content": ""})
    text_msg_idx = len(history) - 1
    
    try:
        # Check database connection
        if not connection.is_usable():
            connection.close()
            connection.connect()
        
        # Status messages for UI feedback
        status_messages = []
        
        # Callback for status updates
        def status_callback(msg: str):
            nonlocal thinking_content, status_messages, current_text_content
            status_messages.append(msg)
            
            if msg.startswith("Thinking:") and mode == "Deepthink":
                part = msg[len("Thinking:"):].lstrip()
                stripped = part.strip()
                
                if stripped and stripped not in thoughts_seen:
                    thoughts_seen.add(stripped)
                    thinking_content += f"{part}\n\n"
            
            # Update UI with status
            history[text_msg_idx]["content"] = f"*{msg}*\n\n{current_text_content}"
            yield history
        
        # Callback for plots
        def plot_callback(fig):
            nonlocal has_plot, plot_fig
            has_plot = True
            plot_fig = fig
            log.info("Plot callback", fig=fig)
            
            # Store the plot globally
            plot_id = f"plot_{len(plot_storage)}"
            plot_storage[plot_id] = fig
            
            # Insert plot message right after the text message
            plot_msg = {"role": "assistant", "content": gr.Plot(value=fig)}
            history.insert(text_msg_idx + 1, plot_msg)
            yield history
        
        # Get the response from coach service
        is_deepthink = mode == "Deepthink"
        
        # Call stream_answer with our stream collector
        final_text = coach_svc.stream_answer(
            query=last_user_msg,
            model=LLModels.GEMINI_25_FLASH,
            stream_box=stream_collector,
            temperature=0.7,
            thinking_budget=0 if mode == "Flash" else None,
            is_deepthink=is_deepthink,
            status_callback=status_callback,
            plot_callback=plot_callback,
        )
        
        # Clear status message and start streaming actual content
        history[text_msg_idx]["content"] = ""
        current_text_content = ""
        
        # Add thinking section if present (for Deepthink mode)
        if thinking_content and mode == "Deepthink":
            thinking_section = f"<details><summary>✨ See what I was thinking...</summary>\n\n{thinking_content}</details>\n\n"
            current_text_content = thinking_section
            history[text_msg_idx]["content"] = current_text_content
            yield history
        
        # Simulate streaming effect for the main content
        chunk_size = 50
        for i in range(0, len(final_text), chunk_size):
            chunk = final_text[:i+chunk_size]
            if thinking_content and mode == "Deepthink":
                history[text_msg_idx]["content"] = f"{thinking_section}{chunk}"
            else:
                history[text_msg_idx]["content"] = chunk
            time.sleep(0.02)  # Small delay for streaming effect
            yield history
        
        # Final update with complete text
        if thinking_content and mode == "Deepthink":
            history[text_msg_idx]["content"] = f"{thinking_section}{final_text}"
        else:
            history[text_msg_idx]["content"] = final_text
        
    except Exception as e:
        log.error("Error in process_bot_response", exc_info=e)
        history[text_msg_idx]["content"] = f"Sorry, I encountered an error: {e}"
    
    yield history

def handle_quick_action(action_key):
    """Handle quick action button clicks"""
    prompt = quick_actions.get(action_key, "")
    if prompt:
        return prompt
    return ""

# Custom CSS
custom_css = """
.chatbot { max-height: 600px; }
.quick-action-btn { 
    width: 100%; 
    margin: 5px 0; 
    padding: 10px;
    text-align: left;
}
.mode-selector {
    margin: 10px 0;
}
details {
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
summary {
    cursor: pointer;
    font-weight: bold;
    color: #4CAF50;
}
details[open] summary {
    margin-bottom: 10px;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="WearM.ai Coach") as demo:
    gr.Markdown("# WearM.ai Coach")
    
    # Initial message
    initial_message = [{
        "role": "assistant",
        "content": """👋 Hi there! I'm your WearM.ai Coach, ready to help you achieve your fitness goals!

I can help you analyze your runs, create training plans, and provide expert guidance on running technique and injury prevention."""
    }]
    
    with gr.Row():
        with gr.Column(scale=4):
            # Chat interface
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                bubble_full_width=False,
                type="messages",
                value=initial_message,
                height=500
            )
            
            # Mode selector
            mode_selector = gr.Radio(
                ["Flash", "Deepthink"],
                value="Flash",
                label="Mode",
                info="Flash for quick responses, Deepthink for complex queries",
                elem_classes=["mode-selector"]
            )
            
            # Multimodal input
            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="multiple",
                placeholder="Type your message or record audio...",
                show_label=False,
                sources=["microphone", "upload"],
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Quick Actions")
            gr.Markdown("Here are some ways we can start:")
            
            # Quick action buttons
            quick_action_btns = []
            for key, prompt in quick_actions.items():
                btn = gr.Button(key, elem_classes=["quick-action-btn"])
                quick_action_btns.append(btn)
                
                # Set up click handler for each button
                btn.click(
                    lambda h, p=prompt: h + [{"role": "user", "content": p}],
                    inputs=[chatbot],
                    outputs=[chatbot]
                ).then(
                    process_bot_response,
                    inputs=[chatbot, mode_selector],
                    outputs=[chatbot]
                ).then(
                    lambda: gr.MultimodalTextbox(interactive=True),
                    outputs=[chat_input]
                )
    
    # Event handlers for chat input
    chat_msg = chat_input.submit(
        add_message,
        inputs=[chatbot, chat_input, mode_selector],
        outputs=[chatbot, chat_input]
    )
    
    # Process bot response
    bot_msg = chat_msg.then(
        process_bot_response,
        inputs=[chatbot, mode_selector],
        outputs=[chatbot]
    )
    
    # Re-enable input after response
    bot_msg.then(
        lambda: gr.MultimodalTextbox(interactive=True),
        outputs=[chat_input]
    )

if __name__ == "__main__":
    demo.launch(share=False)