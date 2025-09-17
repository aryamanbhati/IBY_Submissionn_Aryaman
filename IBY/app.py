# app.py

import streamlit as st
import os
import io
import time
from Agents.parser_agent import parse_resume
from Agents.Question_generator import load_qg_model, generate_questions
from Agents.user_interaction_agent import transcribe_audio
from Agents.feedback_agent import generate_feedback

# --- Page Configuration ---
st.set_page_config(page_title="AI Mock Interview Coach", layout="wide", initial_sidebar_state="expanded")

# --- Initialize Session State ---
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'upload_resume'
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = None
if 'questions' not in st.session_state:
    st.session_state.questions = None
if 'transcribed_answer' not in st.session_state:
    st.session_state.transcribed_answer = None
if 'feedback' not in st.session_state:
    st.session_state.feedback = None
if 'qg_model_loaded' not in st.session_state:
    st.session_state.qg_model_loaded = False
if 'interview_question' not in st.session_state:
    st.session_state.interview_question = None
if 'attempt' not in st.session_state:
    st.session_state.attempt = 1

# --- UI Layout ---
st.title("🤖 AI Mock Interview Coach")
st.markdown("Your personal AI assistant for interview preparation.")
st.divider()

# --- Helper Functions ---
def reset_session():
    """Resets the application to its initial state."""
    keys_to_delete = ['current_step', 'parsed_data', 'questions', 'transcribed_answer', 'feedback', 'interview_question', 'attempt']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.current_step = 'upload_resume'
    st.rerun()

# --- Main Application Flow ---
# Step 1: Resume Upload and Parsing
if st.session_state.current_step == 'upload_resume':
    st.subheader("Step 1: Upload Your Resume 📄")
    uploaded_file = st.file_uploader("Choose a TXT file", type=["txt"])

    if uploaded_file:
        with st.spinner("Parsing resume..."):
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            resume_text = stringio.read()
            st.session_state.parsed_data = parse_resume(resume_text)
        
        st.success("Resume parsed successfully!")
        st.write("---")
        st.info("Extracted Skills:")
        # Use the correct key based on your parser output
        skills = st.session_state.parsed_data.get("skills") or st.session_state.parsed_data.get("technical_skills") or []
        st.write(", ".join(skills))
        
        st.markdown("Ready to start your mock interview?")
        if st.button("Generate Interview Questions", key="generate_q_btn"):
            st.session_state.current_step = 'generate_questions'
            st.rerun()

# Step 2: Question Generation
if st.session_state.current_step == 'generate_questions':
    st.subheader("Step 2: Generating Personalized Questions 💡")

    if not st.session_state.qg_model_loaded:
        with st.spinner("Loading question generation model... This may take a moment."):
            st.session_state.qg_model_loaded = load_qg_model()
    
    if st.session_state.qg_model_loaded:
        with st.spinner("Generating questions based on your skills..."):
            skills = st.session_state.parsed_data.get("skills") or st.session_state.parsed_data.get("technical_skills") or []
            st.session_state.questions = generate_questions(skills)
        
        if st.session_state.questions:
            st.success("Questions generated successfully! You're ready to start.")
            st.session_state.interview_question = st.session_state.questions[st.session_state.attempt - 1]
            st.session_state.current_step = 'conduct_interview'
            st.rerun()
        else:
            st.error("Failed to generate questions. Please try again.")
    else:
        st.error("Question generation model could not be loaded. Please ensure it has been fine-tuned.")

# Step 3: Conduct Interview
if st.session_state.current_step == 'conduct_interview':
    st.subheader(f"Step 3: The Interview (Question {st.session_state.attempt})")
    
    if st.session_state.interview_question:
        st.info(f"**Question:** {st.session_state.interview_question}")
        st.markdown("Please provide your answer by uploading an audio file.")
        
        audio_file = st.file_uploader("Upload an audio file (MP3, WAV)", type=["mp3", "wav"])
        
        if audio_file:
            with st.spinner("Transcribing your answer..."):
                file_path = f"temp_audio_{int(time.time())}.mp3"
                with open(file_path, "wb") as f:
                    f.write(audio_file.getbuffer())
                
                st.session_state.transcribed_answer = transcribe_audio(file_path)
                os.remove(file_path)
            
            st.success("Transcription complete!")
            st.info(f"**Your Answer:** \"{st.session_state.transcribed_answer}\"")
            
            if st.button("Get Feedback", key="get_feedback_btn"):
                st.session_state.current_step = 'get_feedback'
                st.rerun()

# Step 4: Feedback Generation
if st.session_state.current_step == 'get_feedback':
    st.subheader(f"Step 4: AI Feedback for Question {st.session_state.attempt} 🌟")

    if st.session_state.transcribed_answer and st.session_state.interview_question:
        with st.spinner("Generating personalized feedback..."):
            st.session_state.feedback = generate_feedback(
                st.session_state.interview_question, 
                st.session_state.transcribed_answer
            )
        
        st.success("Feedback Generated!")
        st.markdown("---")
        st.markdown(st.session_state.feedback)
        
        st.divider()
        if st.session_state.attempt < len(st.session_state.questions):
            if st.button("Next Question", key="next_q_btn"):
                st.session_state.attempt += 1
                st.session_state.interview_question = st.session_state.questions[st.session_state.attempt - 1]
                st.session_state.transcribed_answer = None
                st.session_state.current_step = 'conduct_interview'
                st.rerun()
        else:
            st.balloons()
            st.success("Congratulations! You have completed the mock interview.")
            if st.button("Start New Interview", key="new_interview_btn"):
                reset_session()
    else:
        st.error("Something went wrong. Please start the interview from the beginning.")
        if st.button("Back to Interview", key="back_to_interview_btn"):
            st.session_state.current_step = 'conduct_interview'
            st.rerun()

# --- Sidebar for Navigation ---
with st.sidebar:
    st.header("Navigation")
    if st.button("Restart Session", key="restart_btn"):
        reset_session()
    
    st.divider()
    st.header("Session Progress")
    progress = (st.session_state.attempt - 1) / len(st.session_state.questions) if st.session_state.questions else 0
    st.progress(progress)
    st.info(f"Question: {st.session_state.attempt} of {len(st.session_state.questions) if st.session_state.questions else 0}")