# app.py

import streamlit as st
from llm_service import GroqLLMService
from speech_to_text import SpeechToTextService

# Initialize services
llm_service = GroqLLMService()
stt_service = SpeechToTextService()

st.set_page_config(page_title="Voice-to-Text LLM Chat")

st.title("🗣️ Voice-to-Text LLM Chat")
st.caption("Powered by Groq and Streamlit")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Voice Input ---
st.sidebar.header("Voice Input")
st.sidebar.markdown("Click 'Start Recording' to speak.")

transcribed_text = stt_service.record_and_transcribe()

if transcribed_text:
    user_input = transcribed_text
    st.sidebar.success(f'Transcribed: "{user_input}"')
else:
    user_input = st.chat_input("Type your message here...")


# --- Process User Input (Text or Voice) ---
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare messages for LLM
    llm_messages = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Get streaming response from LLM
        stream_response = llm_service.get_chat_completion(llm_messages)

        if stream_response:
            for chunk in stream_response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(
                        full_response + "▌"
                    )  # Add blinking cursor

            message_placeholder.markdown(full_response)  # Final message without cursor
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
        else:
            st.error("Failed to get response from LLM.")
