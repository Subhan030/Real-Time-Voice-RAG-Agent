import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

KNOWLEDGE_BASE_DIR = "knowledge_base"
CHROMA_STORE_DIR = "chroma_store"
COLLECTION_NAME = "rag_docs"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

def load_documents(directory: str) -> list[dict]:
    """Load all PDF and TXT files from a directory."""
    docs = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            with pdfplumber.open(filepath) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            docs.append({"source": filename, "text": text})
        elif filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"source": filename, "text": text})
    return docs

def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for i, chunk in enumerate(splits):
            chunks.append({
                "id": f"{doc['source']}_chunk_{i}",
                "text": chunk,
                "source": doc["source"]
            })
    return chunks

def build_index(clear_existing: bool = False):
    """Full pipeline: load → chunk → embed → store."""
    print("[Indexer] Loading documents...")
    docs = load_documents(KNOWLEDGE_BASE_DIR)
    chunks = chunk_documents(docs)
    print(f"[Indexer] {len(chunks)} chunks created from {len(docs)} document(s).")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_STORE_DIR)

    if clear_existing:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[Indexer] Cleared existing collection.")
        except Exception:
            pass

    collection = client.get_or_create_collection(COLLECTION_NAME)
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c["source"]} for c in chunks]
    )
    print("[Indexer] Index built and saved to chroma_store/")

if __name__ == "__main__":
    build_index()

