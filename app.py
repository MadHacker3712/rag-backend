"""
Project 2: RAG Backend API

FastAPI server with:
  GET  /health  — liveness check
  POST /query   — retrieves top-K docs from ChromaDB and returns them

Optimizations:
  - Query result cache (MD5 exact-match) for repeated queries
  - Persistent ChromaDB index (pre-built, loaded once at startup)
  - Embedding model loaded once at startup (no per-request model load)

Run:
  python -m uvicorn app:app --host 0.0.0.0 --port 8000

Test:
  curl -X POST http://localhost:8000/query \
       -H "Content-Type: application/json" \
       -d '{"query": "How do I reset my password?"}'
"""
import hashlib
import os
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Config (override via environment variables) ───────────────────────────────
CHROMA_DIR = os.getenv("CHROMA_DIR", str(Path(__file__).parent / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "customer_support_faq")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "3"))

# ── App State ─────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Customer Support API")

state: dict = {
    "embed_model": None,
    "collection": None,
    "cache": {},
    "cache_hits": 0,
    "cache_misses": 0,
}


# ── Startup: Load models and index once ───────────────────────────────────────
@app.on_event("startup")
def startup():
    print(f"Loading embedding model: {EMBED_MODEL}")
    state["embed_model"] = SentenceTransformer(EMBED_MODEL)

    print(f"Connecting to ChromaDB at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    state["collection"] = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    print(f"RAG API ready. Index has {state['collection'].count()} documents.")


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class RetrievedDoc(BaseModel):
    rank: int
    category: str
    question: str
    answer: str
    distance: float


class QueryResponse(BaseModel):
    query: str
    results: list[RetrievedDoc]
    latency_ms: float
    cache_hit: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_size": state["collection"].count() if state["collection"] else 0,
        "cache_hits": state["cache_hits"],
        "cache_misses": state["cache_misses"],
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t_start = time.perf_counter()
    cache_key = hashlib.md5(req.query.strip().lower().encode()).hexdigest()

    if cache_key in state["cache"]:
        state["cache_hits"] += 1
        latency_ms = (time.perf_counter() - t_start) * 1000
        return QueryResponse(
            query=req.query,
            results=state["cache"][cache_key],
            latency_ms=round(latency_ms, 2),
            cache_hit=True,
        )

    # Cache miss — run full retrieval
    state["cache_misses"] += 1
    embedding = state["embed_model"].encode([req.query])
    docs = state["collection"].query(
        query_embeddings=embedding.tolist(),
        n_results=req.top_k,
        include=["metadatas", "distances"],
    )

    results = [
        RetrievedDoc(
            rank=i + 1,
            category=docs["metadatas"][0][i]["category"],
            question=docs["metadatas"][0][i]["question"],
            answer=docs["metadatas"][0][i]["answer"],
            distance=round(docs["distances"][0][i], 4),
        )
        for i in range(len(docs["metadatas"][0]))
    ]

    state["cache"][cache_key] = results
    latency_ms = (time.perf_counter() - t_start) * 1000

    return QueryResponse(
        query=req.query,
        results=results,
        latency_ms=round(latency_ms, 2),
        cache_hit=False,
    )
