# speech_to_text.py

import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import io
from groq import Groq


class SpeechToTextService:
    def __init__(self):
        """Initializes the Groq client for Whisper API."""
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        except KeyError:
            st.error(
                "GROQ_API_KEY not found in .streamlit/secrets.toml. Please add it."
            )
            st.stop()
        except Exception as e:
            st.error(f"Error initializing Groq client for STT: {e}")
            st.stop()

    def record_and_transcribe(self):
        """
        Records audio from the microphone and transcribes it using Groq's Whisper API.

        Returns:
            str: The transcribed text, or None if no audio was recorded or an error occurred.
        """
        audio_chunk = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            key="mic_recorder",
            callback=None,  # No specific callback needed for immediate transcription
            # use_streamlit_audio=False,  # Use browser's audio for better compatibility
        )

        if audio_chunk:
            try:
                # The mic_recorder returns bytes in WebM format if use_streamlit_audio=False
                # Convert to WAV for broader compatibility with some APIs or if needed
                audio_segment = AudioSegment.from_file(
                    io.BytesIO(audio_chunk["bytes"]), format="webm"
                )
                wav_file_bytes_io = io.BytesIO()
                audio_segment.export(wav_file_bytes_io, format="wav")
                wav_file_bytes_io.seek(0)  # Reset stream position

                # Use Groq's Whisper API for transcription
                with st.spinner("Transcribing audio..."):
                    transcript = self.client.audio.transcriptions.create(
                        file=("audio.wav", wav_file_bytes_io.getvalue()),
                        model="whisper-large-v3",  # Groq's Whisper model
                    )
                return transcript.text
            except Exception as e:
                st.error(f"Error transcribing audio: {e}")
                return None
        return None
