import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a helpful voice assistant. Answer the user's question
using ONLY the provided context. Be concise — your answer will be spoken aloud.
If the answer is not in the context, say: "I don't have that information in my knowledge base."
Do not make up information. Do not use bullet points or markdown formatting."""

def build_prompt(question: str, context: str) -> str:
    return f"""Context:
{context}

Question: {question}

Answer:"""

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
        # Yield complete sentences to TTS as soon as they're ready
        while any(p in buffer for p in [".", "!", "?"]):
            for punct in [".", "!", "?"]:
                idx = buffer.find(punct)
                if idx != -1:
                    sentence = buffer[:idx + 1].strip()
                    buffer = buffer[idx + 1:].lstrip()
                    if sentence:
                        yield sentence
                    break
    if buffer.strip():
        yield buffer.strip()
