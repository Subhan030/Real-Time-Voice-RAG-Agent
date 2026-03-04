# Real-Time Voice RAG Agent

A voice AI agent that listens to your question, retrieves context from a local document
knowledge base, and speaks back a grounded answer — all with low latency.

## Architecture

```
Microphone → faster-whisper (STT) → ChromaDB (RAG) → Groq LLaMA (LLM) → ElevenLabs (TTS) → Speaker
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file:
   ```
   GROQ_API_KEY=your_key_here
   ELEVENLABS_API_KEY=your_key_here   # Optional: falls back to pyttsx3
   ```

3. Add your PDF or TXT files to the `knowledge_base/` folder.

4. Run the agent:
   ```bash
   python main.py
   ```

## Design Decisions

- **Groq + LLaMA 3.1-8b-instant**: Chosen for ~200ms LLM response time.
- **faster-whisper (base model)**: Local STT with good accuracy and low memory use.
- **ChromaDB**: Zero-config persistent vector store, no external server required.
- **Sentence streaming to TTS**: LLM response is streamed and spoken sentence-by-sentence
  to start audio playback before the full response is generated.
- **webrtcvad**: Prevents recording silence, cuts recording the moment the user stops speaking.

## Known Limitations

- Background noise can trigger false recordings — best used in a quiet environment.
- Whisper `base` model may struggle with strong accents; upgrade to `small` if needed.
- ElevenLabs free tier has a monthly character limit.
