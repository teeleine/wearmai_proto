import os
import django
from infrastructure.logging import configure_logging

# Configure Django settings before importing any Django models
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "wearmai.settings")
django.setup()

configure_logging() # Call after django.setup() if it uses Django components

import streamlit as st
from services.llm_coach.coach_service import CoachService
from user_profile.loader import load_profile
from infrastructure.llm_clients.base import LLModels
from django.db import connection
import structlog # Assuming you use structlog for logging

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
    # ... (rest of init_session_state is the same)
    if "user_profile_data" not in st.session_state:
        try:
            if not connection.is_usable():
                connection.close()
                connection.connect()
            st.session_state.user_profile_data = load_profile(name="Test User 2 - Full Data Load")
        except Exception as e:
            st.error(f"Error loading user profile: {str(e)}")
            log.error("Error loading user profile", exc_info=e)
            st.session_state.user_profile_data = None

    if "coach_svc" not in st.session_state:
        if st.session_state.user_profile_data and 'llm_user_profile' in st.session_state.user_profile_data:
            st.session_state.coach_svc = CoachService(
                vs_name="BookChunks_voyage",
                user_profile=st.session_state.user_profile_data['llm_user_profile'],
            )
        elif not st.session_state.user_profile_data:
             pass # Error already shown or handled
        else:
            st.error("User profile loaded but 'llm_user_profile' key is missing.")
            log.error("llm_user_profile missing in user_profile_data", data=st.session_state.user_profile_data)


    if "mode" not in st.session_state:
        st.session_state.mode = "Flash"
    if "show_quick_actions" not in st.session_state:
        st.session_state.show_quick_actions = True
    # Removed selected_action from init, it's managed by button clicks / form submissions

init_session_state()

# Custom CSS
st.markdown("""
    <style>
        .block-container {
            max-width: 80rem; /* Consider slightly less if 80rem is too wide for some screens */
            padding-top: 2rem;
            padding-right: 1rem; /* Keep some padding */
            padding-left: 1rem;  /* Keep some padding */
            padding-bottom: 3rem;
        }
        .stButton button { /* Specific to quick action buttons if needed, or general */
            width: 100%;
            padding: 0.5rem;
            min-height: 3rem; 
            /* margin-bottom: 3rem; */ /* This created large gaps below quick action buttons */
        }
        .stForm { /* Applied to the input form container */
            max-width: 100%; /* Let it use the column width */
        }
        /* Add some margin to the button container for quick actions */
        .quick-action-buttons > div {
            margin-bottom: 1rem; /* Space below quick action rows */
        }
    </style>
""", unsafe_allow_html=True)

st.title("WearM.ai Coach")

# Main chat display container
chat_container = st.container()

def process_message(user_query: str):
    if not st.session_state.get("user_profile_data") or not st.session_state.get("coach_svc"):
        st.error("Unable to process message: System not properly initialized. Please check profile loading and CoachService initialization.")
        st.session_state.messages.append({"role": "assistant", "content": "System not ready. Please try again or contact support."})
        return

    # The user message is already added to st.session_state.messages before calling this function.
    # This function now focuses on generating and adding the assistant's response.

    with st.chat_message("assistant"): # This creates the visual block for the assistant's response
        thinking_expander = None
        # Initialize thoughts_content and thoughts_seen for each call to process_message
        thoughts_content = "••• ⭐ Thinking •••\n\n"
        thoughts_seen = set()
        current_section = None

        is_deepthink_mode = st.session_state.mode == "Deepthink"

        if is_deepthink_mode:
            # Create the expander directly. No need for an st.empty() to hold it here.
            thinking_expander = st.expander("✨ See what I'm thinking...", expanded=False)
            thinking_expander.markdown(thoughts_content, unsafe_allow_html=True)
        
        # This container will hold the streaming final answer
        final_answer_ui_element = st.empty()
        
        with st.status("Processing your request...", expanded=True) as status_box:
            try:
                if not connection.is_usable():
                    status_box.update(label="Reconnecting to database...", state="running")
                    connection.close()
                    connection.connect()

                model_to_use = LLModels.GEMINI_25_FLASH # GEMINI_15_FLASH in original
                
                def update_status_callback(status_msg: str):
                    nonlocal thoughts_content, current_section # Access both variables
                    
                    if status_msg.startswith("Thinking:") and thinking_expander:
                        raw_thought_part = status_msg[len("Thinking:"):].lstrip() # Get text after "Thinking: ", remove leading spaces from this part
                        stripped_thought_part = raw_thought_part.strip()
                        
                        if stripped_thought_part and stripped_thought_part not in thoughts_seen:
                            thoughts_seen.add(stripped_thought_part)
                            
                            # Heuristic: if it doesn't start with typical markdown structural chars or spaces, make it H2
                            if not raw_thought_part.startswith(("#", "*", "-", " ", "\t", ">")) and len(stripped_thought_part.split()) < 10:
                                # This is a new section header
                                current_section = stripped_thought_part
                                thoughts_content = f"••• ⭐ Thinking •••\n\n## {current_section}\n\n"
                            else:
                                # This is content under the current section
                                if current_section:
                                    thoughts_content = f"••• ⭐ Thinking •••\n\n## {current_section}\n\n{raw_thought_part}\n\n"
                                else:
                                    thoughts_content = f"••• ⭐ Thinking •••\n\n{raw_thought_part}\n\n"
                            
                            thinking_expander.markdown(thoughts_content, unsafe_allow_html=True)
                    else:
                        status_box.update(label=status_msg, state="running")
                
                status_box.update(label="Analyzing your request...", state="running")
                final_answer_text = st.session_state.coach_svc.stream_answer(
                    query=user_query,
                    model=model_to_use,
                    stream_box=final_answer_ui_element, # The st.empty() placeholder
                    temperature=0.7,
                    thinking_budget=None,
                    is_deepthink=is_deepthink_mode,
                    status_callback=update_status_callback
                )
                
                if thinking_expander: # Collapse expander if it was used
                    if thoughts_seen: # Only if actual thoughts were added
                        thinking_expander.expanded = False
                    else: # No thoughts shown, make sure it's empty if it only had preamble
                        thinking_expander.markdown("") 
                
                status_box.update(label="Response complete!", state="complete", expanded=False)
                
                assistant_response_data = {"role": "assistant", "content": final_answer_text}
                if is_deepthink_mode and thoughts_seen:
                    # Store cleaned thoughts
                    final_thoughts_md = thoughts_content.replace("••• ⭐ Thinking •••\n\n", "", 1).strip()
                    if final_thoughts_md:
                        assistant_response_data["thoughts_markdown"] = final_thoughts_md
                
                st.session_state.messages.append(assistant_response_data)
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                log.error("Error in process_message", exc_info=e)
                status_box.update(label=error_message, state="error", expanded=True)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                # final_answer_ui_element.error(error_message) # Show error in the answer placeholder

# --- Chat History Display ---
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "thoughts_markdown" in msg and msg["thoughts_markdown"]:
                with st.expander("✨ See what I was thinking...", expanded=False):
                    st.markdown(msg["thoughts_markdown"], unsafe_allow_html=True)
            st.markdown(msg["content"])

# --- Quick Actions ---
quick_actions = {
    "📊 Analyze my last run": "Can you analyze my last run data and provide detailed insights about my performance?",
    "🎯 Create a 4-week plan": "Can you create a personalized 4-week running plan tailored to my current fitness level?",
    "🏃‍♂️ Analyze run technique": "Can you analyze my running technique and provide specific recommendations for improvement?",
    "🚨 Injury risk assessment": "Can you perform a comprehensive injury risk assessment based on my running data and patterns?"
}

if st.session_state.show_quick_actions and len(st.session_state.messages) == 1: # Only initial assistant message
    st.write("Here are some ways we can start:")
    
    # Use a container with a class for specific styling if needed
    with st.container(): # Removed class for simplicity, use st.columns directly
        cols = st.columns(len(quick_actions)) # Dynamically create columns
        for idx, (label, prompt_text) in enumerate(quick_actions.items()):
            if cols[idx].button(label, key=f"quick_action_{idx}", use_container_width=True):
                st.session_state.show_quick_actions = False
                st.session_state.messages.append({"role": "user", "content": prompt_text})
                process_message(prompt_text)
                st.rerun() # Rerun to update display and clear quick actions

# --- Mode Selection ---
mode_col, _ = st.columns([1, 3]) # Use _ for unused spacer
with mode_col:
    selected_mode_option = st.selectbox(
        "Mode",
        ["⚡ Flash (Great for quick questions)", "🧠 Deepthink (Great for complex questions)"],
        index=0 if st.session_state.mode == "Flash" else 1,
        label_visibility="collapsed"
        # Removed help as it's in the option text
    )
    st.session_state.mode = selected_mode_option.split(" ")[1] # "Flash" or "Deepthink"

# --- Input Area ---
# Use st.chat_input for a more conventional chat UI feel if desired,
# or stick to st.form if specific form behaviors are needed.
# Using st.form as per original:
input_container = st.container()
with input_container:
    with st.form(key="message_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])
        with col1:
            user_typed_question = st.text_input(
                "Ask the Coach",
                key="user_text_input_field", # Different key from any state var
                label_visibility="collapsed",
                placeholder="Type your message here..."
            )
        with col2:
            form_submit_button = st.form_submit_button("➤", help="Send message", use_container_width=True)

        if form_submit_button and user_typed_question:
            st.session_state.show_quick_actions = False
            st.session_state.messages.append({"role": "user", "content": user_typed_question})
            process_message(user_typed_question)
            # No st.rerun() here, form submission naturally causes it.
            # The process_message updates st.session_state.messages,
            # and the rerun will render the new state.