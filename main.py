import sys
import os
import threading
import pyaudio
import webrtcvad
from dotenv import load_dotenv

load_dotenv()

from rag.indexer import build_index
from rag.retriever import retrieve
from stt.transcriber import listen_and_transcribe, listen_from_preroll
from llm.generator import generate_stream
from tts.speaker import speak, speak_stream_interruptible

FAISS_STORE_DIR = "faiss_store"

# VAD settings for barge-in monitoring
_SAMPLE_RATE = 16000
_FRAME_MS = 30
_FRAME_SIZE = int(_SAMPLE_RATE * _FRAME_MS / 1000)
_BARGE_IN_FRAMES = 20       # ~600ms of sustained speech to trigger barge-in
_VAD_AGGRESSIVENESS = 3     # 0-3: 3 = strictest speech detection
_ENERGY_THRESHOLD = 300     # RMS amplitude sweet spot (ignores echo, catches real speech)

def ensure_index():
    if not os.path.exists(FAISS_STORE_DIR) or not os.listdir(FAISS_STORE_DIR):
        print("[Main] No index found. Building index from knowledge_base/...")
        build_index()
    else:
        print("[Main] Existing index found. Skipping re-index.")


def barge_in_monitor(interrupt_event: threading.Event, stop_event: threading.Event,
                     preroll_frames: list):
    """Background thread: monitors mic via VAD + energy. Sets interrupt_event if user speaks."""
    import numpy as np
    import time
    # Grace period: ignore the first 350ms so TTS audio burst doesn't self-trigger
    time.sleep(0.35)
    vad = webrtcvad.Vad(_VAD_AGGRESSIVENESS)
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16, channels=1,
        rate=_SAMPLE_RATE, input=True,
        frames_per_buffer=_FRAME_SIZE
    )
    speech_count = 0
    candidate_frames = []
    try:
        while not stop_event.is_set():
            frame = stream.read(_FRAME_SIZE, exception_on_overflow=False)
            # Must pass BOTH energy threshold AND VAD to count as real speech
            rms = np.sqrt(np.mean(np.frombuffer(frame, dtype=np.int16).astype(np.float32) ** 2))
            is_loud_enough = rms > _ENERGY_THRESHOLD
            is_speech = vad.is_speech(frame, _SAMPLE_RATE)
            if is_loud_enough and is_speech:
                speech_count += 1
                candidate_frames.append(frame)  # Collect for pre-roll
                if speech_count >= _BARGE_IN_FRAMES:
                    preroll_frames.extend(candidate_frames)  # Pass audio to STT
                    print("\n[Main] Barge-in detected — interrupting agent.")
                    interrupt_event.set()
                    break
            else:
                speech_count = 0
                candidate_frames.clear()  # Reset if silence detected
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def speak_response(question: str):
    """Retrieve context, generate answer, speak with barge-in. Returns preroll if interrupted."""
    context = retrieve(question)
    print("[Main] Generating answer...")
    sentence_stream = generate_stream(question, context)

    interrupt_event = threading.Event()
    stop_monitor = threading.Event()
    preroll_frames = []
    monitor_thread = threading.Thread(
        target=barge_in_monitor,
        args=(interrupt_event, stop_monitor, preroll_frames),
        daemon=True
    )
    monitor_thread.start()
    speak_stream_interruptible(sentence_stream, interrupt_event)
    stop_monitor.set()
    monitor_thread.join(timeout=1.0)

    return preroll_frames if interrupt_event.is_set() else None


def run():
    ensure_index()
    speak("Hello! I'm ready. Ask me anything about the documents.")
    print("\n[Main] Voice RAG Agent running. Speak to interrupt at any time. Ctrl+C to exit.\n")

    while True:
        try:
            # Step 1: Listen
            question = listen_and_transcribe()
            if not question:
                continue

            # Step 2: Retrieve + generate + speak (with barge-in)
            print("[Main] Retrieving context...")
            preroll = speak_response(question)

            # If interrupted, immediately listen using pre-roll frames then respond again
            if preroll is not None:
                print("[Main] Interrupted — listening immediately...")
                question2 = listen_from_preroll(preroll) if preroll else listen_and_transcribe()
                if question2:
                    print("[Main] Retrieving context...")
                    speak_response(question2)

        except KeyboardInterrupt:
            print("\n[Main] Shutting down. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"[Main] Error: {e}")
            speak("Sorry, something went wrong. Please try again.")


if __name__ == "__main__":
    run()
