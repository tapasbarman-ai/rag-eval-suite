import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

DATA_FILE = PROJECT_ROOT / "data" / "data.txt"
DB_DIR = PROJECT_ROOT / "chroma_store"
COLLECTION_NAME = "research_paper"


# 1. EMBEDDINGS ---- Gemini embedding function for Chroma
class GeminiEmbeddings(EmbeddingFunction):
    def __init__(self, model: str = "gemini-embedding-001"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        all_embeddings = []
        batch_size = 50
        for i in range(0, len(input), batch_size):
            batch = input[i : i + batch_size]
            response = self.client.models.embed_content(
                model=self.model,
                contents=batch
            )
            all_embeddings.extend([item.values for item in response.embeddings])
        return all_embeddings


# 2. LOAD & CHUNK ---- read data.txt and split into overlapping chunks
def load_and_chunk(chunk_size_words: int = 200, overlap_words: int = 50):
    with open(DATA_FILE, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    words = text.split()
    chunks = []
    step = chunk_size_words - overlap_words

    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + chunk_size_words]))
        if i + chunk_size_words >= len(words):
            break

    return chunks


# 3. BUILD ---- embed once, keep on disk so we don't re-embed
def load_store():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    embed_fn = GeminiEmbeddings()

    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
        if collection.count() > 0:
            return collection

    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    chunks = load_and_chunk()

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.basename(DATA_FILE), "chunk_index": i} for i in range(len(chunks))]

    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    return collection


# 4. RETRIEVER ---- query top-k most relevant chunks
def retrieve(query: str, top_k: int = 5):
    collection = load_store()
    results = collection.query(query_texts=[query], n_results=top_k)

    return [
        {"content": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]


# 5. TRY IT ---- python src/retriver.py
if __name__ == "__main__":
    results = retrieve("What is Multi-Head Attention and how does it work?", top_k=2)

    for r in results:
        print(f"[Chunk {r['metadata']['chunk_index']}] {r['content'][:150]}...\n")