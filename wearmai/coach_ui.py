import os
import django
from infrastructure.logging import configure_logging
import plotly.express as px
import plotly.io  # Add this import for optional JSON serialization
from services.speech.speech_service import SpeechService

# Configure Django settings before importing any Django models
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "wearmai.settings")
django.setup()

configure_logging()  # Call after django.setup() if it uses Django components

import streamlit as st
from services.llm_coach.coach_service import CoachService
from user_profile.loader import load_profile
from infrastructure.llm_clients.base import LLModels
from django.db import connection
import structlog  # Assuming you use structlog for logging

log = structlog.get_logger(__name__)

# Set page config first
st.set_page_config(layout="wide")


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """👋 Hi there! I'm your WearM.ai Coach, ready to help you achieve your fitness goals!

I can help you analyze your runs, create training plans, and provide expert guidance on running technique and injury prevention."""
            }
        ]

    if "user_profile_data" not in st.session_state:
        try:
            # Ensure database connection is fresh
            connection.close()
            connection.connect()
            
            # Load profile with retry mechanism
            max_retries = 3
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    st.session_state.user_profile_data = load_profile(name="Test User 2 - Full Data Load")
                    break
                except ValueError as ve:
                    # Handle specific ValueError from QuerySet evaluation
                    log.warning("QuerySet evaluation error, retrying...", exc_info=ve)
                    connection.close()
                    connection.connect()
                    retry_count += 1
                    last_error = ve
                    if retry_count == max_retries:
                        raise ve
                except Exception as e:
                    log.error("Unexpected error loading user profile", exc_info=e)
                    last_error = e
                    break
            
            if last_error:
                st.error(f"Error loading user profile: {last_error}")
                st.session_state.user_profile_data = None
                
        except Exception as e:
            st.error(f"Error loading user profile: {e}")
            log.error("Error loading user profile", exc_info=e)
            st.session_state.user_profile_data = None

    if "coach_svc" not in st.session_state:
        upd = st.session_state.user_profile_data
        if upd and "llm_user_profile" in upd:
            st.session_state.coach_svc = CoachService(
                vs_name="Bookchunks",
                user_profile=upd["llm_user_profile"],
            )
        elif not upd:
            pass  # already shown error
        else:
            st.error("User profile loaded but 'llm_user_profile' key is missing.")
            log.error("llm_user_profile missing", data=upd)

    if "mode" not in st.session_state:
        st.session_state.mode = "Flash"
    if "show_quick_actions" not in st.session_state:
        st.session_state.show_quick_actions = True

    if "speech_service" not in st.session_state:
        st.session_state.speech_service = SpeechService()

    # Reset audio-related states
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "pending_transcription" not in st.session_state:
        st.session_state.pending_transcription = ""
    if "last_audio_rec" not in st.session_state:
        st.session_state.last_audio_rec = None
    if "audio_processing" not in st.session_state:
        st.session_state.audio_processing = False
    # Add a counter to force audio input reset
    if "audio_input_key" not in st.session_state:
        st.session_state.audio_input_key = 0


init_session_state()

# Custom CSS
st.markdown(
    """
    <style>
      .block-container { max-width:80rem; padding:2rem 1rem 3rem; }
      .stButton button { width:100%; padding:0.5rem; min-height:3rem; }
      .stForm { max-width:100%; }
      .stForm > div { margin-bottom: 0; }
      .stTextInput > div > div > input {
        padding: 10px 15px;
        background-color: rgb(240, 242, 246);
      }
      .stSelectbox {
        max-width: 19rem;
        padding-top: 5rem;
      }
      span[data-testid="stAudioInputWaveformTimeCode"] {
        display: none;
      }
      /* Add styles for audio input container */
      div[data-testid="stAudioInput"] {
        max-width: 90px;
      }
      .quick-action-buttons > div { margin-bottom:1rem; }
      .microphone-button { 
        border: none;
        background: none;
        cursor: pointer;
        font-size: 1.5rem;
        padding: 0.5rem;
        border-radius: 50%;
        transition: background-color 0.3s;
      }
      .microphone-button:hover {
        background-color: rgba(0, 0, 0, 0.1);
      }
      .microphone-button.recording {
        color: red;
        animation: pulse 1.5s infinite;
      }
      @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
      }
      .transcribed-text {
        background-color: #f0f8ff;
        border: 1px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        font-style: italic;
      }

      /* NEW: Reduce gap and width of audio column */
      .stHorizontalBlock:has(div[data-testid="stAudioInput"]) {
        gap: 0rem !important;
      }
      
      /* Force the audio column to be much narrower */
      .stColumn:has(div[data-testid="stAudioInput"]) {
        max-width: 100px !important;
        min-width: 100px !important;
        flex: 0 0 100px !important;
        border-radius: 0px;
        border: none;
      }

      .stAudioInput > div:first-of-type {
        border-radius: 0px;
        border: none;
      }
      
      /* Make the vertical block inside audio column narrower */
      .stColumn:has(div[data-testid="stAudioInput"]) .stVerticalBlock {
        max-width: 120px !important;
      }
      
      /* Ensure chat column takes remaining space */
      .stColumn:has(.stForm) {
        flex: 1 1 auto !important;
      }
      .stForm {
        border-radius: 0px;
        background-color: rgb(240, 242, 246);
        border: none;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("WearM.ai Coach")

def process_message(user_query: str):
    if not st.session_state.get("user_profile_data") or not st.session_state.get("coach_svc"):
        st.error("System not ready. Please try again later.")
        st.session_state.messages.append(
            {"role": "assistant", "content": "System not ready. Please contact support."}
        )
        return

    # Add the user message to session state if not already present
    if not st.session_state.messages or st.session_state.messages[-1]["role"] != "user":
        st.session_state.messages.append({"role": "user", "content": user_query})

    # show the assistant bubble
    with chat_container:
        with st.chat_message("assistant"):
            thinking_expander = None
            thoughts_placeholder = None
            thoughts_content = ""
            thoughts_seen = set()
            current_section = None
            is_deep = st.session_state.mode == "Deepthink"
            has_shown_initial_message = False

            if is_deep:
                thinking_expander = st.expander("✨ See what I'm thinking...", expanded=False)
                thoughts_placeholder = thinking_expander.empty()

            # Create placeholders for plot and text
            plot_placeholder = st.empty()  # Plot will appear here first
            final_answer_ui = st.empty()   # Text will stream below it

            with st.status("Processing your request...", expanded=True) as status_box:
                try:
                    if not connection.is_usable():
                        status_box.update(label="Reconnecting to database...", state="running")
                        connection.close()
                        connection.connect()

                    def status_cb(msg: str):
                        nonlocal thoughts_content, current_section, has_shown_initial_message
                        if msg.startswith("Thinking:") and thoughts_placeholder:
                            part = msg[len("Thinking:") :].lstrip()
                            stripped = part.strip()
                            
                            # Skip empty or already seen content
                            if not stripped or stripped in thoughts_seen:
                                return
                                
                            # Add to seen set immediately to prevent duplicates
                            thoughts_seen.add(stripped)
                            
                            # Show initial message only before first thought
                            if not has_shown_initial_message:
                                thoughts_content = "⭐ The model's thoughts will be shown below\n\n"
                                has_shown_initial_message = True

                            # Check if this looks like a section header
                            if (
                                not part.startswith(("#", "*", "-", " ", "\t", ">"))
                                and len(stripped.split()) < 10
                            ):
                                current_section = stripped
                                thoughts_content += f"## {current_section}\n\n"
                            else:
                                thoughts_content += f"{part}\n\n"
                                
                            # Update the placeholder with new content instead of the expander
                            thoughts_placeholder.markdown(thoughts_content, unsafe_allow_html=True)
                        else:
                            status_box.update(label=msg, state="running")

                    # Initialize message data early
                    data = {"role": "assistant"}
                    msg_idx = len(st.session_state.messages)

                    status_box.update(label="Analyzing your request...", state="running")
                    final_text = st.session_state.coach_svc.stream_answer(
                        query=user_query,
                        model=LLModels.GEMINI_25_FLASH,
                        stream_box=final_answer_ui,
                        temperature=0.7,
                        thinking_budget=0 if st.session_state.mode == "Flash" else None,
                        is_deepthink=is_deep,
                        status_callback=status_cb,
                        # Add plot callback that shows and stores the figure
                        plot_callback=lambda fig: (
                            plot_placeholder.plotly_chart(fig, use_container_width=True),
                            st.session_state.__setitem__(f"fig_{msg_idx}", fig.to_json()),
                            data.__setitem__("plot_key", f"fig_{msg_idx}")
                        ),
                    )

                    data["content"] = final_text

                    # FIXED: Only modify the expander if we have thoughts to show
                    if thinking_expander and thoughts_seen:
                        thinking_expander.expanded = False
                        # Store the cleaned thoughts for the message history
                        cleaned = thoughts_content.replace("⭐ The model's thoughts will be shown below\n\n", "", 1).strip()
                        if cleaned:
                            data["thoughts_markdown"] = cleaned

                    status_box.update(label="Response complete!", state="complete", expanded=False)

                    st.session_state.messages.append(data)

                except Exception as e:
                    log.error("Error in process_message", exc_info=e)
                    err = f"Sorry, I encountered an error: {e}"
                    status_box.update(label=err, state="error", expanded=True)
                    st.session_state.messages.append({"role": "assistant", "content": err})

# Container for the chat bubbles - MOVED TO TOP
chat_container = st.container()

# ----- Render existing chat history -----
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # 1. Plot first (if present and from assistant)
            if msg["role"] == "assistant" and msg.get("plot_key"):
                fig_json = st.session_state.get(msg["plot_key"])
                if fig_json:
                    fig = plotly.io.from_json(fig_json)
                    st.plotly_chart(fig, use_container_width=True)

            # 2. Expandable 'thoughts' section
            if msg["role"] == "assistant" and msg.get("thoughts_markdown"):
                with st.expander("✨ See what I was thinking...", expanded=False):
                    st.markdown(msg["thoughts_markdown"], unsafe_allow_html=True)

            # 3. Main answer text
            st.markdown(msg["content"])

# ----- Quick Actions -----
quick_actions = {
    "📊 Analyze my last run": "Can you analyze my last run data and provide detailed insights about my performance?",
    "🎯 Create a 4-week plan": "Can you create a personalized 4-week running plan tailored to my current fitness level?",
    "🏃‍♂️ Analyze run technique": "Can you analyze my running technique and provide specific recommendations for improvement?",
    "🚨 Injury risk assessment": "Can you perform a comprehensive injury risk assessment based on my running data and patterns?",
}

# Create a container for quick actions that we can hide immediately
quick_actions_placeholder = st.empty()

# Only show quick actions if they're enabled and it's the first message
if st.session_state.show_quick_actions and len(st.session_state.messages) == 1:
    with quick_actions_placeholder:
        st.write("Here are some ways we can start:")
        cols = st.columns(len(quick_actions))
        for i, (lbl, prompt) in enumerate(quick_actions.items()):
            if cols[i].button(lbl, key=f"qa_{i}", use_container_width=True):
                # Hide quick actions immediately
                quick_actions_placeholder.empty()
                st.session_state.show_quick_actions = False
                # Add and show user message immediately
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                process_message(prompt)
                st.rerun()  # Force a clean rerun after processing

# ----- Mode Selection and Chat Input -----
# Mode dropdown at the top
new_mode = st.selectbox(
    "Mode",
    ["Flash", "Deepthink"],
    index=0 if st.session_state.mode == "Flash" else 1,
    format_func=lambda x: f"{'⚡' if x == 'Flash' else '🧠'} {x} ({'quick responses' if x == 'Flash' else 'complex queries'})",
    label_visibility="collapsed"
)

if new_mode != st.session_state.mode:
    st.session_state.mode = new_mode
    st.rerun()

# Audio input and chat input in a row below
audio_col, chat_col = st.columns([1, 3], gap="small")

with audio_col:
    # Audio input
    audio_data = st.audio_input(
        "🎤", 
        label_visibility="collapsed", 
        key=f"audio_rec_{st.session_state.audio_input_key}"
    )

with chat_col:
    # Create a form to handle the input with pre-populated text
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([10, 1], gap="small")
        
        with col1:
            # Use text_input with the transcribed text as default value
            user_input = st.text_input(
                "Message", 
                value=st.session_state.pending_transcription,
                placeholder="Type a message or click the mic to record audio...",
                label_visibility="collapsed",
                key="message_input"
            )
        
        with col2:
            send_clicked = st.form_submit_button("➤", use_container_width=True)

        # Handle form submission
        if send_clicked and user_input.strip():
            # Clear any pending transcription
            st.session_state.pending_transcription = ""
            # Hide quick actions immediately
            quick_actions_placeholder.empty()
            st.session_state.show_quick_actions = False
            # Store the message for next script run
            st.session_state.messages.append({"role": "user", "content": user_input})
            # Flag that a message is waiting to be processed
            st.session_state.to_process = True
            # Force an immediate rerun to clear the input
            st.rerun()

# ----- Process message after rerun if there's a new message to process -----
if st.session_state.get("to_process", False):
    # Reset the processing flag
    st.session_state.to_process = False
    # Process the message - no need to show it again since it's already in chat history
    process_message(st.session_state.messages[-1]["content"])
    st.rerun()  # Force a clean rerun to update the UI

# Has the user just recorded something new?
new_audio = (
    audio_data is not None
    and not st.session_state.audio_processing
)

if new_audio:
    # Set processing flag to prevent multiple transcriptions
    st.session_state.audio_processing = True
    
    # Show a temporary processing message
    with st.spinner("Transcribing audio..."):
        # Transcribe the audio
        transcription = st.session_state.speech_service.transcribe_audio(
            audio_data.getvalue(), "wav"
        )

    # Store the transcription to populate the chat input
    st.session_state.pending_transcription = transcription
    # Increment the key to force audio input reset on next render
    st.session_state.audio_input_key += 1
    # Reset processing flag
    st.session_state.audio_processing = False

    # Rerun to show the transcribed text in the chat input
    st.rerun()