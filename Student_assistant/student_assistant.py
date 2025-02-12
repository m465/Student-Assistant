import streamlit as st
import whisper
import os
import tempfile
import google.generativeai as genai
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Load Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

model = load_model()

# Set up API key securely
genai.configure(api_key=os.getenv("AIzaSyBYTITnkXaYIMPGBMbyrniKLAZJx0bu4k4"))  # Ensure the API key is set as an env variable

# Initialize memory for conversation
memory = ConversationSummaryMemory(llm=ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest"), return_messages=True)

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest")
conversation = ConversationChain(llm=llm, memory=memory)

# Function to get AI-powered response
def get_gemini_explanation(query):
    response = conversation.predict(input=query)
    return response if response else "Error retrieving explanation."

# UI Layout
st.title("🎙️ AI-Powered Study Assistant")
st.write("Record your question or type it, and get AI-powered explanations.")

# Use Streamlit columns to create side-by-side layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎤 Record Your Question")

    recorder_html = """
        <script>
        let mediaRecorder;
        let audioChunks = [];
        let recording = false;

        function startRecording() {
            navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                recording = true;
                document.getElementById('recording-status').innerText = "🔴 Recording...";

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    let audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    let audioUrl = URL.createObjectURL(audioBlob);
                    document.getElementById('audio-playback').src = audioUrl;
                    document.getElementById('audio-playback').style.display = "block";

                    let downloadLink = document.getElementById('audio-download');
                    downloadLink.href = audioUrl;
                    downloadLink.download = "recorded_audio.wav";
                    downloadLink.style.display = "block";

                    document.getElementById('recording-status').innerText = "✅ Recording Stopped. Ready for Playback.";
                };
            });
        }

        function stopRecording() {
            mediaRecorder.stop();
        }
        </script>

        <button onclick="startRecording()">🎤 Start Recording</button>
        <button onclick="stopRecording()">⏹️ Stop Recording</button>
        <p id="recording-status">⚪ Not Recording</p>
        <audio id="audio-playback" controls style="display:none;"></audio>
        <br>
        <a id="audio-download" style="display:none;">⬇️ Download Audio</a>
    """
    st.components.v1.html(recorder_html, height=180)

with col2:
    st.subheader("⌨️ Type Your Question")
    user_input = st.text_area("Enter your question here:", height=100)

    if user_input:
        st.write("**Your Question:**", user_input)
        explanation = get_gemini_explanation(user_input)
        st.subheader("📘 AI-Powered Explanation")
        st.write(explanation)
