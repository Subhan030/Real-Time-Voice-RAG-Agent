import sys
import os
from dotenv import load_dotenv

load_dotenv()

from rag.indexer import build_index
from rag.retriever import retrieve
from stt.transcriber import listen_and_transcribe
from llm.generator import generate_stream
from tts.speaker import speak, speak_stream

CHROMA_STORE_DIR = "chroma_store"

def ensure_index():
    """Build the vector index if it doesn't already exist."""
    if not os.path.exists(CHROMA_STORE_DIR) or not os.listdir(CHROMA_STORE_DIR):
        print("[Main] No index found. Building index from knowledge_base/...")
        build_index()
    else:
        print("[Main] Existing index found. Skipping re-index.")

def run():
    ensure_index()
    speak("Hello! I'm ready. Ask me anything about the documents.")
    print("\n[Main] Voice RAG Agent is running. Press Ctrl+C to exit.\n")

    while True:
        try:
            # Step 1: Listen and transcribe
            question = listen_and_transcribe()
            if not question:
                continue

            # Step 2: Retrieve relevant context
            print("[Main] Retrieving context...")
            context = retrieve(question)

            # Step 3: Generate and stream answer to TTS
            print("[Main] Generating answer...")
            sentence_stream = generate_stream(question, context)
            speak_stream(sentence_stream)

        except KeyboardInterrupt:
            print("\n[Main] Shutting down. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"[Main] Error: {e}")
            speak("Sorry, something went wrong. Please try again.")

if __name__ == "__main__":
    run()
