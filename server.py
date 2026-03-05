import os
import json
import base64
import asyncio
import tempfile
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import aiofiles

from rag.indexer import build_index
from rag.retriever import retrieve, reload_collection
from stt.transcriber import transcribe_audio_bytes
from llm.generator import generate_stream
from tts.speaker import generate_audio

CHROMA_STORE_DIR = "chroma_store"
KNOWLEDGE_BASE_DIR = "knowledge_base"

app = FastAPI(title="Voice RAG Agent")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def ensure_index():
    if not os.path.exists(CHROMA_STORE_DIR) or not os.listdir(CHROMA_STORE_DIR):
        print("[Server] No index found. Building index from knowledge_base/...")
        build_index()
    else:
        print("[Server] Existing index found. Skipping re-index.")


@app.on_event("startup")
async def startup():
    ensure_index()
    print("[Server] Voice RAG Agent server is ready.")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the knowledge base and re-index."""
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
    filepath = os.path.join(KNOWLEDGE_BASE_DIR, file.filename)
    async with aiofiles.open(filepath, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Re-index using ChromaDB API (not filesystem delete)
    build_index(clear_existing=True)
    reload_collection()

    return {"status": "ok", "message": f"Uploaded {file.filename} and re-indexed."}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[WS] Client connected.")

    interrupted = asyncio.Event()

    async def send_json(data: dict):
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            pass

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "audio_data":
                interrupted.clear()

                audio_b64 = msg.get("audio", "")
                audio_bytes = base64.b64decode(audio_b64)

                await send_json({"type": "status", "state": "processing"})

                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, transcribe_audio_bytes, audio_bytes)

                if not text:
                    await send_json({"type": "status", "state": "listening"})
                    continue

                await send_json({"type": "transcript", "text": text})

                context = await loop.run_in_executor(None, retrieve, text)

                await send_json({"type": "status", "state": "speaking"})

                def _generate_sentences(question, ctx):
                    return list(generate_stream(question, ctx))

                sentences = await loop.run_in_executor(
                    None, _generate_sentences, text, context
                )

                for sentence in sentences:
                    if interrupted.is_set():
                        print("[WS] Response interrupted by user.")
                        break

                    await send_json({
                        "type": "answer_text",
                        "text": sentence,
                        "done": False
                    })

                    audio_bytes_out, fmt = await loop.run_in_executor(
                        None, generate_audio, sentence
                    )
                    audio_b64_out = base64.b64encode(audio_bytes_out).decode("utf-8")
                    await send_json({
                        "type": "answer_audio",
                        "audio": audio_b64_out,
                        "format": fmt,
                        "sentence": sentence
                    })

                if not interrupted.is_set():
                    await send_json({"type": "answer_text", "text": "", "done": True})

                await send_json({"type": "status", "state": "listening"})

            elif msg_type == "interrupt":
                print("[WS] Interrupt received from client.")
                interrupted.set()

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[WS] Error: {e}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
