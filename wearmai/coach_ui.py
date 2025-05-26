import os
import django
from infrastructure.logging import configure_logging

# Configure Django settings before importing any Django models
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "wearmai.settings")
django.setup()

configure_logging()

import streamlit as st
from services.llm_coach.coach_service import CoachService
from user_profile.loader import load_profile
from infrastructure.llm_clients.base import LLModels
from django.db import connection

# Set page config first before any other Streamlit commands
st.set_page_config(layout="wide")

def init_session_state():
    """Initialize all session state variables"""
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
            # Ensure database connection is ready
            if not connection.is_usable():
                connection.close()
                connection.connect()
            st.session_state.user_profile_data = load_profile(name="Test User 2 - Full Data Load")
        except Exception as e:
            st.error(f"Error loading user profile: {str(e)}")
            st.session_state.user_profile_data = None

    if "coach_svc" not in st.session_state:
        if st.session_state.user_profile_data:
            st.session_state.coach_svc = CoachService(
                vs_name="BookChunks_voyage",
                user_profile=st.session_state.user_profile_data['llm_user_profile'],
            )

    if "mode" not in st.session_state:
        st.session_state.mode = "Flash"

    if "show_quick_actions" not in st.session_state:
        st.session_state.show_quick_actions = True

    if "selected_action" not in st.session_state:
        st.session_state.selected_action = None

# Initialize session state
init_session_state()

# Custom CSS for wider content and better spacing
st.markdown("""
    <style>
        .block-container {
            max-width: 80rem;
            padding-top: 2rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 3rem;
        }
        .stButton button {
            width: 100%;
            padding: 0.5rem;
            min-height: 3rem;
            margin-bottom: 3rem;
        }
        .stForm {
            max-width: 50rem !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("WearM.ai Coach")

# Create main chat container
chat_container = st.container()

def process_message(message: str):
    """Helper function to process messages consistently"""
    if not st.session_state.user_profile_data or not st.session_state.coach_svc:
        st.error("Unable to process message: System not properly initialized")
        return

    st.session_state.messages.append({"role": "user", "content": message})
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(message)
        
        with st.chat_message("assistant"):
            final_answer_container = st.empty()
            
            with st.status("Processing your request...", expanded=True) as status_box:
                try:
                    # Ensure database connection is ready
                    if not connection.is_usable():
                        status_box.update(label="Reconnecting to database...", state="running")
                        connection.close()
                        connection.connect()

                    model = LLModels.GEMINI_25_FLASH if st.session_state.mode == "Flash" else LLModels.GEMINI_25_PRO
                    thinking_budget = 0 if st.session_state.mode == "Flash" else 30
                    
                    def update_status(status_msg: str):
                        status_box.update(label=status_msg, state="running")
                    
                    status_box.update(label="Analyzing your request...", state="running")
                    final_answer_text = st.session_state.coach_svc.stream_answer(
                        query=message,
                        model=model,
                        stream_box=final_answer_container,
                        temperature=0.7,
                        thinking_budget=thinking_budget,
                        status_callback=update_status
                    )
                    
                    status_box.update(label="Response complete!", state="complete", expanded=False)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer_text})
                    
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    status_box.update(label=error_message, state="error")
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    st.error(error_message)
                    st.exception(e)

# Display chat history in the container
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Quick actions
quick_actions = {
    "📊 Analyze my last run": "Can you analyze my last run data and provide detailed insights about my performance?",
    "🎯 Create a 4-week plan": "Can you create a personalized 4-week running plan tailored to my current fitness level?",
    "🏃‍♂️ Analyze run technique": "Can you analyze my running technique and provide specific recommendations for improvement?",
    "🚨 Injury risk assessment": "Can you perform a comprehensive injury risk assessment based on my running data and patterns?"
}

# Check if we need to process a selected action
if st.session_state.selected_action:
    prompt = quick_actions[st.session_state.selected_action]
    st.session_state.selected_action = None
    st.session_state.show_quick_actions = False
    process_message(prompt)
    st.rerun()

# Only show quick actions if no conversation has started
elif st.session_state.show_quick_actions and len(st.session_state.messages) == 1:
    st.write("Here are some ways we can start:")
    
    # Create a container with padding for the buttons
    button_container = st.container()
    with button_container:
        # Use columns with less spacing between them
        cols = st.columns([1, 1, 1, 1])
        for idx, (label, _) in enumerate(quick_actions.items()):
            with cols[idx]:
                if st.button(label, key=f"quick_action_{idx}", use_container_width=True):
                    st.session_state.selected_action = label
                    st.rerun()

# Mode selection dropdown with more width
mode_col, spacer = st.columns([1, 3])
with mode_col:
    st.session_state.mode = st.selectbox(
        "Mode",
        ["⚡ Flash (Great for quick questions)", "🧠 Deepthink (Great for complex questions)"],
        index=0 if st.session_state.mode == "Flash" else 1,
        label_visibility="collapsed",
        help="Great for quick questions and simple analyses"
    )
    # Strip emojis for internal use
    st.session_state.mode = st.session_state.mode.split(" ")[1]

# Input area with text input and send button
input_container = st.container()
with input_container:
    # Use a form to handle both Enter key and button click
    with st.form(key="message_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])  # Wider input field
        with col1:
            question = st.text_input(
                "Ask the Coach",
                key="user_input",
                label_visibility="collapsed",
                placeholder="Type your message here..."
            )
        with col2:
            submit_button = st.form_submit_button("➤", help="Send message", use_container_width=True)

        if submit_button and question:
            st.session_state.show_quick_actions = False
            process_message(question)