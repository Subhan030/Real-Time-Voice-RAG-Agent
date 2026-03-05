from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_STORE_DIR = "chroma_store"
COLLECTION_NAME = "rag_docs"
TOP_K = 4

_model = SentenceTransformer("all-MiniLM-L6-v2")
_client = chromadb.PersistentClient(path=CHROMA_STORE_DIR)
_collection = _client.get_or_create_collection(COLLECTION_NAME)

def reload_collection():
    """Reload the collection reference after re-indexing."""
    global _collection
    _collection = _client.get_or_create_collection(COLLECTION_NAME)


def retrieve(query: str) -> str:
    """Embed query and return top-k relevant chunks as a single context string."""
    query_embedding = _model.encode([query]).tolist()
    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K
    )
    chunks = results["documents"][0]
    context = "\n\n---\n\n".join(chunks)
    return context
