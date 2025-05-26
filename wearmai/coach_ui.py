import os
import django
from infrastructure.logging import configure_logging

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
            if not connection.is_usable():
                connection.close()
                connection.connect()
            st.session_state.user_profile_data = load_profile(name="Test User 2 - Full Data Load")
        except Exception as e:
            st.error(f"Error loading user profile: {e}")
            log.error("Error loading user profile", exc_info=e)
            st.session_state.user_profile_data = None

    if "coach_svc" not in st.session_state:
        upd = st.session_state.user_profile_data
        if upd and "llm_user_profile" in upd:
            st.session_state.coach_svc = CoachService(
                vs_name="BookChunks_voyage",
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


init_session_state()

# Custom CSS
st.markdown(
    """
    <style>
      .block-container { max-width:80rem; padding:2rem 1rem 3rem; }
      .stButton button { width:100%; padding:0.5rem; min-height:3rem; }
      .stForm { max-width:100%; }
      .quick-action-buttons > div { margin-bottom:1rem; }
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

    # show the assistant bubble
    with chat_container:  # Use the chat container for assistant messages
        with st.chat_message("assistant"):
            thinking_expander = None
            thoughts_content = "••• ⭐ Thinking •••\n\n"
            thoughts_seen = set()
            current_section = None
            is_deep = st.session_state.mode == "Deepthink"

            if is_deep:
                thinking_expander = st.expander("✨ See what I'm thinking...", expanded=False)
                thinking_expander.markdown(thoughts_content, unsafe_allow_html=True)

            final_answer_ui = st.empty()

            with st.status("Processing your request...", expanded=True) as status_box:
                try:
                    if not connection.is_usable():
                        status_box.update(label="Reconnecting to database...", state="running")
                        connection.close()
                        connection.connect()

                    def status_cb(msg: str):
                        nonlocal thoughts_content, current_section
                        if msg.startswith("Thinking:") and thinking_expander:
                            part = msg[len("Thinking:") :].lstrip()
                            stripped = part.strip()
                            if stripped and stripped not in thoughts_seen:
                                thoughts_seen.add(stripped)
                                if (
                                    not part.startswith(("#", "*", "-", " ", "\t", ">"))
                                    and len(stripped.split()) < 10
                                ):
                                    current_section = stripped
                                    thoughts_content = f"••• ⭐ Thinking •••\n\n## {current_section}\n\n"
                                else:
                                    if current_section:
                                        thoughts_content = (
                                            f"••• ⭐ Thinking •••\n\n## {current_section}\n\n"
                                            + part
                                            + "\n\n"
                                        )
                                    else:
                                        thoughts_content = f"••• ⭐ Thinking •••\n\n{part}\n\n"
                                thinking_expander.markdown(thoughts_content, unsafe_allow_html=True)
                        else:
                            status_box.update(label=msg, state="running")

                    status_box.update(label="Analyzing your request...", state="running")
                    final_text = st.session_state.coach_svc.stream_answer(
                        query=user_query,
                        model=LLModels.GEMINI_25_FLASH,
                        stream_box=final_answer_ui,
                        temperature=0.7,
                        thinking_budget=None,
                        is_deepthink=is_deep,
                        status_callback=status_cb,
                    )

                    if thinking_expander:
                        if thoughts_seen:
                            thinking_expander.expanded = False
                        else:
                            thinking_expander.markdown("")

                    status_box.update(label="Response complete!", state="complete", expanded=False)

                    data = {"role": "assistant", "content": final_text}
                    if is_deep and thoughts_seen:
                        cleaned = thoughts_content.replace("••• ⭐ Thinking •••\n\n", "", 1).strip()
                        if cleaned:
                            data["thoughts_markdown"] = cleaned

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
            if msg["role"] == "assistant" and msg.get("thoughts_markdown"):
                with st.expander("✨ See what I was thinking...", expanded=False):
                    st.markdown(msg["thoughts_markdown"], unsafe_allow_html=True)
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
                quick_actions_placeholder.empty()
                st.session_state.show_quick_actions = False
                # Add and show user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                process_message(prompt)
                st.rerun()  # Force a clean rerun after processing

# ----- Mode Selection ----- MOVED ABOVE CHAT INPUT
mode_col, _ = st.columns([1, 3])
with mode_col:
    sel = st.selectbox(
        "Mode",
        ["⚡ Flash (quick)", "🧠 Deepthink (complex)"],
        index=0 if st.session_state.mode == "Flash" else 1,
        label_visibility="collapsed",
    )
    st.session_state.mode = sel.split(" ")[1]

# Chat input using st.chat_input - MOVED TO BOTTOM
if user_message := st.chat_input("Ask the Coach"):
    # Immediately hide quick actions
    quick_actions_placeholder.empty()
    st.session_state.show_quick_actions = False
    # Add and show user message
    st.session_state.messages.append({"role": "user", "content": user_message})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_message)
    process_message(user_message)
    st.rerun()  # Force a clean rerun after processing