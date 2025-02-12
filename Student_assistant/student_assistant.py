import streamlit as st
import whisper
import os
import tempfile
import google.generativeai as genai
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Load Whisper Tiny model for faster speech-to-text processing
@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

model = load_model()

# Set up API key securely
genai.configure(api_key=os.getenv("AIzaSyBYTITnkXaYIMPGBMbyrniKLAZJx0bu4k4"))  # Set this in Streamlit Cloud

# Initialize memory for conversation
memory = ConversationSummaryMemory(llm=ChatGoogleGenerativeAI(model="gemini-pro-vision"), return_messages=True)

# Initialize Gemini smaller model
llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
conversation = ConversationChain(llm=llm, memory=memory)

# Function to get AI-powered response
def get_gemini_explanation(query):
    response = conversation.predict(input=query)
    return response if response else "Error retrieving explanation."

# Function to process audio and get text
def transcribe_audio(audio_file):
    try:
        transcript = model.transcribe(audio_file)
        return transcript["text"]
    except Exception as e:
        return f"Error transcribing audio: {e}"

# UI Layout
st.title("🎙️ AI-Powered Study Assistant")
st.write("Record your question or type it, and get AI-powered explanations.")

# Use Streamlit columns to create side-by-side layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎤 Record or Upload Audio")

    # Upload audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

    # Record audio directly
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio_path = temp_audio.name

        st.audio(temp_audio_path, format="audio/wav")

        # Transcribe audio
        transcribed_text = transcribe_audio(temp_audio_path)
        st.write("**Transcribed Text:**", transcribed_text)

        # Get AI response
        explanation = get_gemini_explanation(transcribed_text)
        st.subheader("📘 AI-Powered Explanation")
        st.write(explanation)

with col2:
    st.subheader("⌨️ Type Your Question")
    user_input = st.text_area("Enter your question here:", height=100)

    if user_input:
        st.write("**Your Question:**", user_input)
        explanation = get_gemini_explanation(user_input)
        st.subheader("📘 AI-Powered Explanation")
        st.write(explanation)
