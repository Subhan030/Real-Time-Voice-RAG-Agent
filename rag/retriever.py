from sentence_transformers import SentenceTransformer
import os
import faiss
import pickle
import numpy as np

FAISS_STORE_DIR = "faiss_store"
TOP_K = 4

_model = SentenceTransformer("all-MiniLM-L6-v2")

# We use global variables to hold the loaded index and chunks
_index = None
_chunks = None

def _load_faiss_store():
    """Lazily load the index and chunks."""
    global _index, _chunks
    if _index is None or _chunks is None:
        index_path = os.path.join(FAISS_STORE_DIR, "index.faiss")
        chunks_path = os.path.join(FAISS_STORE_DIR, "chunks.pkl")
        
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            _index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                _chunks = pickle.load(f)
        else:
            _index = None
            _chunks = []

def retrieve(query: str) -> str:
    """Embed query and return top-k relevant chunks as a single context string using FAISS."""
    _load_faiss_store()
    
    if _index is None or not _chunks:
        return ""
        
    query_embedding = _model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    
    # Search for top TOP_K nearest neighbors
    distances, indices = _index.search(query_embedding, TOP_K)
    
    retrieved_texts = []
    for idx in indices[0]:
        if idx != -1 and idx < len(_chunks):
            retrieved_texts.append(_chunks[idx]["text"])
            
    context = "\n\n---\n\n".join(retrieved_texts)
    return context
