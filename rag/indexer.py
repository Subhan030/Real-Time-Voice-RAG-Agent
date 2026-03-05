import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

KNOWLEDGE_BASE_DIR = "knowledge_base"
FAISS_STORE_DIR = "faiss_store"
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
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Create FAISS Store directory
    os.makedirs(FAISS_STORE_DIR, exist_ok=True)
    
    # Initialize and build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    
    # Save the index to disk
    index_path = os.path.join(FAISS_STORE_DIR, "index.faiss")
    faiss.write_index(index, index_path)
    
    # Save the chunk dictionaries (metadata and text)
    chunks_path = os.path.join(FAISS_STORE_DIR, "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
        
    print("[Indexer] Index built and saved to faiss_store/")

if __name__ == "__main__":
    build_index()

