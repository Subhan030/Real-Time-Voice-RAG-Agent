# Real-Time Voice RAG Agent 🎙️

A completely local-first, low-latency conversational AI agent that combines Speech-to-Text (STT), Retrieval-Augmented Generation (RAG), Large Language Models (LLM), and Text-to-Speech (TTS) into a seamless, interruptible voice experience.

## ✨ Key Features

1. **Real-Time Voice Interruption (Barge-in):**
   - The agent listens while speaking. If you interrupt it mid-sentence, it instantly stops playback and starts transcribing your new question.
   - Built with WebRTC Voice Activity Detection (VAD) and RMS energy thresholds to ignore background noise and its own speaker echo.
   - **Pre-roll Buffer:** Captures the first milliseconds of your interruption so no words are "swallowed" during the VAD detection phase.

2. **Local-First Knowledge Base (RAG):**
   - Drop any `.txt`, `.md`, or `.pdf` files into the `knowledge_base/` folder.
   - The system automatically parses PDFs (via `pdfplumber`), chunks the text, and builds a local **ChromaDB** vector index.
   - When you speak, the agent retrieves the most relevant paragraphs and uses them to answer, aggressively preventing hallucination on out-of-domain questions.

3. **Ultra-Low Latency Pipeline:**
   - **STT (Whisper Tiny):** Transcribes your speech locally in milliseconds as soon as you stop talking.
   - **LLM (Groq API):** Uses the blazing-fast `llama-3.3-70b-versatile` model with token streaming.
   - **TTS (ElevenLabs & pyttsx3):** Features a parallel producer-consumer pipeline. Sentence `N+1` is fetched from ElevenLabs while sentence `N` is actively playing through your speakers, eliminating dead air between sentences.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A [Groq API Key](https://console.groq.com/)
- An [ElevenLabs API Key](https://elevenlabs.io/) (Optional, falls back to local `pyttsx3` if missing)
- PortAudio (for PyAudio mic access):
  - macOS: `brew install portaudio`
  - Linux: `sudo apt-get install portaudio19-dev`

### Installation

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Add your API keys to a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=gsk_your_groq_key_here
   ELEVENLABS_API_KEY=optional_elevenlabs_key_here
   ```

3. Add your context documents to the `knowledge_base/` folder.

### Running the Agent

Simply run the main script:
```bash
python main.py
```

1. Look for the `[STT] Listening... (speak now)` prompt.
2. Ask a question about your documents.
3. While the agent is responding, **speak over it** to test the barge-in interruption!

## 🧠 Architecture Overview

- `main.py`: The central orchestrator loop (listen → retrieve → generate → speak) and the background VAD interruption monitor.
- `stt/transcriber.py`: Uses `pyaudio`, `webrtcvad`, and `faster-whisper` (`tiny` model) to detect silence and transcribe speech locally.
- `rag/indexer.py` & `rag/retriever.py`: Chunks documents using `RecursiveCharacterTextSplitter` and embeds/stores them using `sentence-transformers` and `ChromaDB`. 
- `llm/generator.py`: Streams prompt-injected LLM responses from Groq, yielding text chunk-by-chunk at sentence boundaries.
- `tts/speaker.py`: Converts text to lifelike audio using the ElevenLabs streaming API, played sequentially via `pygame` on a background thread.
