import pyaudio
import webrtcvad
import numpy as np
import wave
import tempfile
import os
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30          # VAD frame size (10, 20, or 30ms only)
SILENCE_TIMEOUT_FRAMES = 50     # ~1.5 seconds of silence before stopping
VAD_AGGRESSIVENESS = 2          # 0–3: higher = more aggressive filtering
WHISPER_MODEL_SIZE = "tiny"     # Options: tiny, base, small, medium

_vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
_whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

def record_until_silence() -> bytes:
    """Record from microphone, return raw PCM bytes after user stops speaking."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
    )

    print("[STT] Listening... (speak now)")
    frames = []
    silence_count = 0
    speaking_started = False

    while True:
        frame = stream.read(int(SAMPLE_RATE * FRAME_DURATION_MS / 1000), exception_on_overflow=False)
        is_speech = _vad.is_speech(frame, SAMPLE_RATE)

        if is_speech:
            speaking_started = True
            silence_count = 0
            frames.append(frame)
        elif speaking_started:
            silence_count += 1
            frames.append(frame)
            if silence_count > SILENCE_TIMEOUT_FRAMES:
                break

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("[STT] Recording complete.")
    return b"".join(frames)

def transcribe(audio_bytes: bytes) -> str:
    """Write audio bytes to a temp WAV file and transcribe with Whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)

    segments, _ = _whisper.transcribe(tmp_path, beam_size=1)
    text = " ".join(seg.text for seg in segments).strip()
    os.unlink(tmp_path)
    print(f"[STT] Transcribed: {text}")
    return text

def listen_and_transcribe() -> str:
    """Combined convenience function: record → transcribe → return text."""
    audio = record_until_silence()
    return transcribe(audio)
