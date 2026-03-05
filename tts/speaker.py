import os
import io
import threading
import pyttsx3
import pygame
from dotenv import load_dotenv

load_dotenv()

USE_ELEVENLABS = bool(os.getenv("ELEVENLABS_API_KEY"))

# Initialize pygame mixer for audio playback
pygame.mixer.init()

def _play_via_pygame(audio_stream, fmt, interrupt_event=None):
    """Load audio into pygame and play, stopping immediately if interrupt_event fires."""
    pygame.mixer.music.load(audio_stream, fmt)
    pygame.mixer.music.play()
    pygame.time.wait(150)  # Buffer to avoid clipping first words
    while pygame.mixer.music.get_busy():
        if interrupt_event and interrupt_event.is_set():
            pygame.mixer.music.stop()
            return
        pygame.time.wait(50)

# --- ElevenLabs TTS (if API key present) ---
def _speak_elevenlabs(text: str, interrupt_event=None):
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    audio = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_turbo_v2",
        output_format="mp3_44100_128"
    )
    audio_bytes = b"".join(audio) if hasattr(audio, '__iter__') and not isinstance(audio, bytes) else audio
    _play_via_pygame(io.BytesIO(audio_bytes), "mp3", interrupt_event)

# --- pyttsx3 TTS (offline fallback) ---
_pyttsx3_engine = pyttsx3.init()
_pyttsx3_engine.setProperty("rate", 175)   # Words per minute
_pyttsx3_engine.setProperty("volume", 0.9)

# Warm up pyttsx3 so the first real sentence isn't clipped by engine init delay
_pyttsx3_engine.say(" ")
_pyttsx3_engine.runAndWait()

def _speak_pyttsx3(text: str, interrupt_event=None):
    import tempfile
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        _pyttsx3_engine.save_to_file(", " + text, tmp)
        _pyttsx3_engine.runAndWait()
        _play_via_pygame(tmp, "wav", interrupt_event)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

# --- Public interface ---
def speak(text: str, interrupt_event=None):
    """Speak text using best available TTS engine."""
    print(f"[TTS] Speaking: {text}")
    if USE_ELEVENLABS:
        try:
            _speak_elevenlabs(text, interrupt_event)
        except Exception as e:
            print(f"[TTS] ElevenLabs failed ({e}), falling back to pyttsx3.")
            _speak_pyttsx3(text, interrupt_event)
    else:
        _speak_pyttsx3(text, interrupt_event)

def stop_speaking():
    """Immediately stop any playing TTS audio."""
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass
    try:
        _pyttsx3_engine.stop()
    except Exception:
        pass


def speak_stream(sentence_generator):
    """Accept a generator of sentences and speak each one as it arrives."""
    for sentence in sentence_generator:
        speak(sentence)


def speak_stream_interruptible(sentence_generator, interrupt_event):
    """
    Parallel TTS pipeline: generate audio for sentence N+1 while playing N.
    Eliminates the dead gap between sentences for natural conversational flow.
    """
    import queue as _queue

    audio_q = _queue.Queue(maxsize=3)

    def _producer():
        for sentence in sentence_generator:
            if interrupt_event.is_set():
                break
            print(f"[TTS] Speaking: {sentence}")
            try:
                audio_bytes, fmt = generate_audio(sentence)
                audio_q.put((audio_bytes, fmt))
            except Exception as e:
                print(f"[TTS] Error generating audio: {e}")
        audio_q.put(None)  # sentinel

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    while True:
        try:
            item = audio_q.get(timeout=10)
        except _queue.Empty:
            break
        if item is None or interrupt_event.is_set():
            break
        audio_bytes, fmt = item
        _play_via_pygame(io.BytesIO(audio_bytes) if isinstance(audio_bytes, bytes) else audio_bytes,
                         fmt, interrupt_event)

    producer_thread.join(timeout=1.0)





def generate_audio(text: str) -> tuple:
    """Generate TTS audio bytes for sending to browser. Returns (bytes, format_str)."""
    if USE_ELEVENLABS:
        try:
            from elevenlabs.client import ElevenLabs
            client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
            audio = client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_turbo_v2",
                output_format="mp3_44100_128"
            )
            audio_bytes = b"".join(audio) if hasattr(audio, '__iter__') and not isinstance(audio, bytes) else audio
            return audio_bytes, "mp3"
        except Exception as e:
            print(f"[TTS] ElevenLabs failed ({e}), falling back to pyttsx3.")

    import tempfile as _tempfile
    tmp_path = _tempfile.mktemp(suffix=".wav")
    try:
        _pyttsx3_engine.save_to_file(text, tmp_path)
        _pyttsx3_engine.runAndWait()
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        return audio_bytes, "wav"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

