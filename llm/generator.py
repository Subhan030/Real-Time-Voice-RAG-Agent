import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a voice assistant. Your ONLY job is to answer questions based on the DOCUMENT CONTENT provided below.
- Answer concisely — your reply will be spoken aloud.
- Use ONLY information from the DOCUMENT CONTENT section. Never invent facts.
- If the answer is not found in the DOCUMENT CONTENT, say exactly: "I don't have that information in my knowledge base."
- Do NOT describe yourself or your own capabilities. If asked "what is the document about?", summarize the DOCUMENT CONTENT, not yourself."""

def build_prompt(question: str, context: str) -> str:
    return f"""DOCUMENT CONTENT:
--- start ---
{context}
--- end ---

User question: {question}

Answer based only on the DOCUMENT CONTENT above:"""

def generate(question: str, context: str) -> str:
    """Call Groq LLM with grounded prompt. Returns full response string."""
    prompt = build_prompt(question, context)
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.3,
        stream=False
    )
    answer = response.choices[0].message.content.strip()
    print(f"[LLM] Answer: {answer}")
    return answer

def generate_stream(question: str, context: str):
    """Stream response sentence by sentence for low-latency TTS."""
    prompt = build_prompt(question, context)
    stream = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.3,
        stream=True
    )

    buffer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        buffer += delta
        # Yield on sentence-ending punctuation
        while any(p in buffer for p in [".", "!", "?"]):
            for punct in [".", "!", "?"]:
                idx = buffer.find(punct)
                if idx != -1:
                    sentence = buffer[:idx + 1].strip()
                    buffer = buffer[idx + 1:].lstrip()
                    if sentence:
                        yield sentence
                    break
        # Also yield at commas/clauses if buffer is long — starts TTS sooner
        if len(buffer) > 80 and "," in buffer:
            idx = buffer.rfind(",")
            phrase = buffer[:idx + 1].strip()
            buffer = buffer[idx + 1:].lstrip()
            if phrase:
                yield phrase
    if buffer.strip():
        yield buffer.strip()
