# agents/user_interaction_agent.py
import whisper

WHISPER_MODEL_SIZE = "base"

try:
    stt_model = whisper.load_model(WHISPER_MODEL_SIZE)
    print(f"Whisper '{WHISPER_MODEL_SIZE}' model loaded successfully. ✅")
except Exception as e:
    print(f"Error loading Whisper model: {e} ❌")
    print("Please ensure you have the necessary dependencies installed.")
    stt_model = None

def transcribe_audio(audio_file_path):
    """
    Transcribes an audio file to text using the Whisper model.

    Args:
        audio_file_path (str): The path to the audio file (e.g., MP3, WAV).

    Returns:
        str: The transcribed text. Returns an error message if transcription fails.
    """
    if stt_model is None:
        return "Error: Speech-to-text model is not available."

    try:
        result = stt_model.transcribe(audio_file_path)
        return result.get("text", "")
    except Exception as e:
        print(f"Transcription failed: {e}")

        return f"Transcription failed: {e}"
