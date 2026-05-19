# Project 2: RAG Backend Optimization

Fine-tuning retrieval speed for a production customer support system.

## What is RAG?

**Retrieval-Augmented Generation** — instead of relying only on the LLM's training data, you search a knowledge base first. This lets you answer questions about *your* products without retraining the model.

## Architecture

```
User Query
    │
    ▼
FastAPI /query (app.py)
    │
    ├──[MD5 cache HIT]──► Return cached results (~12ms)
    │
    └──[MISS]
         │
         ▼
    SentenceTransformer encode()
    (all-MiniLM-L6-v2 → 384-dim vector)
         │
         ▼
    ChromaDB HNSW index query
    (cosine similarity, top-K nearest docs)
         │
         ▼
    Store in cache (MD5 key → results)
         │
         ▼
    Return JSON results + latency_ms + cache_hit
         │
         └── Append to logs/rag.jsonl
```

---

## Architecture

```
project2_rag/
├── data/faq.json          # 30 customer support FAQ entries
├── build_index.py         # Embeds FAQ docs → stores in ChromaDB
├── benchmark.py           # Baseline vs cached retrieval comparison
├── app.py                 # FastAPI endpoint (GET /health, POST /query)
├── Dockerfile             # Pre-builds index at image build time
└── results/
    └── rag_benchmark.json # Benchmark output
```

**Embedding model**: `all-MiniLM-L6-v2` (22MB, 384-dim vectors, fast CPU inference)  
**Vector DB**: ChromaDB with cosine similarity, HNSW index  
**Knowledge base**: 30 customer support FAQ entries across 5 categories

---

## Benchmark Results

*20 queries, 30% repeat rate (6 cache hits), `all-MiniLM-L6-v2`, CPU*

| Setup | Embed (ms) | Retrieve (ms) | Total (ms) | Throughput (QPS) | Cache Hit Rate |
|---|---:|---:|---:|---:|---:|
| Baseline (no cache) | 52.6 | 11.4 | 64.1 | 15.6 | 0% |
| Cached (MD5 exact-match) | 30.1 | 5.4 | 35.6 | 28.1 | 30% |
| **Improvement** | | | **-44.5%** | **+80.3%** | |

**Key insight**: With only 30% repeat queries, caching cuts latency by 44% and nearly doubles throughput. In production systems with higher repeat rates (common in customer support — same questions daily), hit rates of 60–80% are typical, giving even larger gains.

---

## How It Works

### Embedding-based Retrieval

Every FAQ entry is converted to a 384-dimensional vector using `all-MiniLM-L6-v2`. When a user query arrives, the query is embedded the same way, and ChromaDB finds the top-K closest FAQ entries using cosine similarity.

This means "How do I change my password?" and "I forgot my password" both retrieve the same answer — even though the exact words differ. That's the power of semantic search over keyword search.

### Query Cache

```python
cache_key = hashlib.md5(query.strip().lower().encode()).hexdigest()

if cache_key in cache:
    return cache[cache_key]   # ~0ms — no embedding, no DB call

result = embed(query) → search(chromadb) → top_docs
cache[cache_key] = result
return result
```

Cache is exact-match on normalized query text. A production system would add:
- **TTL** (expire cache entries after N hours to stay fresh)
- **Semantic cache** (cache hits for queries that *mean* the same thing, not just identical text)
- **Persistent cache** (Redis) instead of in-memory dict

### ChromaDB HNSW Index

ChromaDB uses **HNSW** (Hierarchical Navigable Small World) — a graph-based approximate nearest neighbor algorithm. Query time is O(log N) instead of O(N), which means:
- 30 docs: ~11ms
- 30,000 docs: ~20ms (barely slower)
- 3,000,000 docs: ~30ms

This scales far better than brute-force search.

---

## How to Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Build the ChromaDB index (run once):**
```bash
python build_index.py
```

**Run benchmark:**
```bash
python benchmark.py
```

**Start the API server:**
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**Test the API:**
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "How do I reset my password?"}'
```

**Check health + cache stats:**
```bash
curl http://localhost:8000/health
```

---

## Docker

```bash
# Build (pre-builds ChromaDB index inside the image)
docker build -t rag-backend:v1 .

# Run
docker run -p 8000:8000 rag-backend:v1
```

The Dockerfile pre-builds the ChromaDB index at image build time using `RUN python build_index.py`. This means:
- Container starts in seconds (no index build on first request)
- Index is reproducible — same data, same embeddings, every time

---

## Trade-offs

| | Exact-match Cache | Semantic Cache | No Cache |
|---|---|---|---|
| **Implementation** | 5 lines | Complex (second vector DB) | None |
| **Hit rate (typical)** | 20–40% | 50–80% | 0% |
| **Staleness risk** | Low (add TTL) | Medium | None |
| **Memory overhead** | Low | High | None |
| **Best for** | High repeat rate queries | Varied phrasings of same question | Development |

---

## Connection to Project 1

In Project 1, the bottleneck was **LLM generation** (4,500ms per response).  
In Project 2, retrieval is only **64ms** — fast enough that the bottleneck shifts back to generation if you add an LLM.

A production pipeline combining both:
```
Query (64ms retrieval) → Context + DialoGPT-small (4,500ms generation) = ~4,600ms total
Query (cached, 0ms retrieval) → Context + DialoGPT-small (4,500ms generation) = ~4,500ms total
```

For a real production system you'd use vLLM on GPU (from Week 5-6 study) to bring generation to ~50ms, making the full pipeline ~100ms end-to-end.
