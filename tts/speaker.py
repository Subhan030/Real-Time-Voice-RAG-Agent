import os
import io
import pyttsx3
import pygame
from dotenv import load_dotenv

load_dotenv()

USE_ELEVENLABS = bool(os.getenv("ELEVENLABS_API_KEY"))

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# --- ElevenLabs TTS (if API key present) ---
def _speak_elevenlabs(text: str):
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    audio = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",   # Replace with your preferred voice ID
        model_id="eleven_turbo_v2",          # Fastest ElevenLabs model
        output_format="mp3_44100_128"
    )
    # Collect generator output into bytes and play with pygame
    audio_bytes = b"".join(audio) if hasattr(audio, '__iter__') and not isinstance(audio, bytes) else audio
    audio_stream = io.BytesIO(audio_bytes)
    pygame.mixer.music.load(audio_stream, "mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)

# --- pyttsx3 TTS (offline fallback) ---
_pyttsx3_engine = pyttsx3.init()
_pyttsx3_engine.setProperty("rate", 175)   # Words per minute
_pyttsx3_engine.setProperty("volume", 0.9)

def _speak_pyttsx3(text: str):
    _pyttsx3_engine.say(text)
    _pyttsx3_engine.runAndWait()

# --- Public interface ---
def speak(text: str):
    """Speak text using best available TTS engine."""
    print(f"[TTS] Speaking: {text}")
    if USE_ELEVENLABS:
        try:
            _speak_elevenlabs(text)
        except Exception as e:
            print(f"[TTS] ElevenLabs failed ({e}), falling back to pyttsx3.")
            _speak_pyttsx3(text)
    else:
        _speak_pyttsx3(text)

def speak_stream(sentence_generator):
    """Accept a generator of sentences and speak each one as it arrives."""
    for sentence in sentence_generator:
        speak(sentence)
