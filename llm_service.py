# llm_service.py

import os
from groq import Groq
import streamlit as st


class GroqLLMService:
    def __init__(self):
        """Initializes the Groq client with the API key."""
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        except KeyError:
            st.error(
                "GROQ_API_KEY not found in .streamlit/secrets.toml. Please add it."
            )
            st.stop()
        except Exception as e:
            st.error(f"Error initializing Groq client: {e}")
            st.stop()

    def get_chat_completion(self, messages, model="llama3-8b-8192", stream=True):
        """
        Sends a list of messages to the Groq LLM and returns the completion.

        Args:
            messages (list): A list of message dictionaries (e.g., {"role": "user", "content": "Hello"}).
            model (str): The Groq model to use (default: "llama3-8b-8192").
            stream (bool): Whether to stream the response (default: True).

        Returns:
            Generator or object: A generator if stream=True, otherwise a completion object.
        """
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=model,
                stream=stream,
            )
            return response
        except Exception as e:
            st.error(f"Error communicating with Groq LLM: {e}")
            return None
