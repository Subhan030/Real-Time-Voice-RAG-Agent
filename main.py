import sys
import os
import threading
import pyaudio
import webrtcvad
from dotenv import load_dotenv

load_dotenv()

from rag.indexer import build_index
from rag.retriever import retrieve
from stt.transcriber import listen_and_transcribe
from llm.generator import generate_stream
from tts.speaker import speak, speak_stream_interruptible

CHROMA_STORE_DIR = "chroma_store"

# VAD settings for barge-in monitoring
_SAMPLE_RATE = 16000
_FRAME_MS = 30
_FRAME_SIZE = int(_SAMPLE_RATE * _FRAME_MS / 1000)
_BARGE_IN_FRAMES = 15       # ~450ms of sustained speech to trigger barge-in
_VAD_AGGRESSIVENESS = 3     # 0-3: higher = stricter speech detection
_ENERGY_THRESHOLD = 300     # RMS amplitude (0-32768); raise to require louder speech

def ensure_index():
    if not os.path.exists(CHROMA_STORE_DIR) or not os.listdir(CHROMA_STORE_DIR):
        print("[Main] No index found. Building index from knowledge_base/...")
        build_index()
    else:
        print("[Main] Existing index found. Skipping re-index.")


def barge_in_monitor(interrupt_event: threading.Event, stop_event: threading.Event):
    """Background thread: monitors mic via VAD + energy. Sets interrupt_event if user speaks."""
    import numpy as np
    vad = webrtcvad.Vad(_VAD_AGGRESSIVENESS)
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16, channels=1,
        rate=_SAMPLE_RATE, input=True,
        frames_per_buffer=_FRAME_SIZE
    )
    speech_count = 0
    try:
        while not stop_event.is_set():
            frame = stream.read(_FRAME_SIZE, exception_on_overflow=False)
            # Must pass BOTH energy threshold AND VAD to count as real speech
            rms = np.sqrt(np.mean(np.frombuffer(frame, dtype=np.int16).astype(np.float32) ** 2))
            is_loud_enough = rms > _ENERGY_THRESHOLD
            is_speech = vad.is_speech(frame, _SAMPLE_RATE)
            if is_loud_enough and is_speech:
                speech_count += 1
                if speech_count >= _BARGE_IN_FRAMES:
                    print("\n[Main] Barge-in detected — interrupting agent.")
                    interrupt_event.set()
                    break
            else:
                speech_count = 0
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


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

            # Step 2: Retrieve
            print("[Main] Retrieving context...")
            context = retrieve(question)

            # Step 3: Generate + stream to TTS with barge-in support
            print("[Main] Generating answer...")
            sentence_stream = generate_stream(question, context)

            interrupt_event = threading.Event()
            stop_monitor = threading.Event()
            monitor_thread = threading.Thread(
                target=barge_in_monitor,
                args=(interrupt_event, stop_monitor),
                daemon=True
            )
            monitor_thread.start()

            speak_stream_interruptible(sentence_stream, interrupt_event)

            # Signal monitor thread to stop
            stop_monitor.set()
            monitor_thread.join(timeout=1.0)

            if interrupt_event.is_set():
                print("[Main] Interrupted — listening immediately...")
                # Loop back to listen right away (no delay)

        except KeyboardInterrupt:
            print("\n[Main] Shutting down. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"[Main] Error: {e}")
            speak("Sorry, something went wrong. Please try again.")


if __name__ == "__main__":
    run()

