import os
import django
from infrastructure.logging import configure_logging
configure_logging()


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "wearmai.settings")
django.setup()

import streamlit as st
from services.llm_coach.coach_service import CoachService
from user_profile.loader import load_profile
from infrastructure.llm_clients.base import LLModels

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_profile_data" not in st.session_state:
    st.session_state.user_profile_data = load_profile(name="Test User 2 - Full Data Load")

if "coach_svc" not in st.session_state:
    st.session_state.coach_svc = CoachService(
        vs_name="BookChunks_voyage",
        user_profile=st.session_state.user_profile_data['llm_user_profile'],
    )

st.title("WearM.ai Coach")

# Display full chat history (keep as is)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.chat_input("Ask the Coach")

if question:
    # Add user message to chat history immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Prepare container for the final assistant response
    with st.chat_message("assistant"):
        final_answer_container = st.empty() # Placeholder where the final answer will stream

        # Use st.status to show progress during context retrieval and generation
        with st.status("Thinking...", expanded=True) as status_box: # Start expanded

            # Define the callback function to update the status box label
            def status_update_callback(message):
                status_box.update(label=message)

            try:

                # 1. Update status before final generation
                status_update_callback("Generating response...")

                # 2. Stream the final answer using the retrieved context
                # Pass the final_answer_container to the streaming function
                final_answer_text = st.session_state.coach_svc.stream_answer(
                    query=question,
                    model=LLModels.GEMINI_25_FLASH,
                    stream_box=final_answer_container,
                    temperature=0.7,
                    thinking_budget=0
                )

                # 3. Update status upon completion
                status_box.update(label="Request complete!", state="complete", expanded=False)

                # 4. Add the complete assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": final_answer_text})

            except Exception as e:
                status_box.update(label=f"An error occurred: {e}", state="error", expanded=True)
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}"})
                final_answer_container.error(f"Sorry, I encountered an error: {e}") # Show error in main chat too
                st.exception(e) # Log the full exception