# Case Study: Building a Production RAG Backend — Architecture, Caching, and Monitoring

*By MadHacker3712 | May 2026*

---

## What Is RAG and Why Does It Matter?

Large language models have a knowledge cutoff. They're also expensive to fine-tune every time your product data changes. Retrieval-Augmented Generation (RAG) solves this by giving the model a live lookup mechanism: before generating a response, retrieve the most relevant documents from a knowledge base, then pass them to the model as context.

For a customer support system, this is ideal. The FAQ changes weekly. You don't retrain the model — you update the vector database.

But the retrieval layer is where most production RAG systems fall apart. Slow retrieval means slow end-to-end latency. Bloated indexes mean high memory costs. No monitoring means silent failures you find out about from user complaints.

This case study walks through the architecture decisions I made, the benchmark results, and the monitoring setup that makes it production-safe.

---

## Architecture

```
User query
    │
    ▼
FastAPI /query endpoint (app.py)
    │
    ├── MD5 cache lookup ──► [HIT]  ──► Return cached results (12ms avg)
    │
    └── [MISS]
         │
         ▼
    SentenceTransformer embed_query()
    (all-MiniLM-L6-v2, 384-dim vectors)
         │
         ▼
    ChromaDB HNSW index query
    (cosine similarity, top-K results)
         │
         ▼
    Return top-K FAQ entries
    + log to logs/rag.jsonl
         │
         ▼
    Cache result (MD5 key → results)
```

Three layers do all the work:
1. **Embedding model** — converts the user's query into a 384-dimensional vector
2. **Vector database** — ChromaDB with HNSW index finds the nearest stored FAQ vectors
3. **Cache** — exact-match MD5 cache catches repeated queries before they hit the embedding model

---

## Implementation Decisions

### Why ChromaDB?

I evaluated three options: FAISS (Facebook AI Similarity Search), ChromaDB, and Milvus.

- **FAISS** is fast but requires manual persistence. You rebuild the index from scratch on restart.
- **Milvus** is the production choice at scale — Kubernetes-native, distributed — but operationally heavy for a single-server deployment.
- **ChromaDB** gives persistent storage (SQLite + HNSW via hnswlib) with zero infrastructure. PersistentClient writes to disk on every insert. On restart, the index loads in milliseconds.

For a portfolio project or small-to-medium deployment (<1M documents), ChromaDB wins on simplicity.

### Why all-MiniLM-L6-v2?

This model is 22MB and runs in ~45ms on CPU. It produces 384-dimensional vectors with strong semantic similarity performance on retrieval benchmarks (MTEB). For a customer support FAQ with short questions and answers, semantic similarity matters more than raw length — "I can't log in" and "password reset" need to match.

A larger model (e.g. `all-mpnet-base-v2`, 420MB) would improve recall slightly, but at 8-10× the inference cost. Not worth it at this scale.

### MD5 Cache

Customer support queries repeat. "How do I reset my password?" gets asked constantly. Running the full embedding + HNSW retrieval for every identical query wastes CPU.

The cache key is the MD5 hash of the lowercased, stripped query string. The cache lives in memory (a Python dict on the `app` state object). This means:
- Zero extra latency on cache hits (memory lookup only)
- Cache is cleared on process restart (intentionally — stale results are never served)
- No separate infrastructure needed (no Redis, no Memcached)

For higher cache durability, you'd add Redis with a TTL. For this deployment, in-memory is correct.

---

## Benchmark Results

I tested 20 queries (30% repeated — realistic for a support queue) across baseline and cached configurations:

| Configuration | Avg Latency | Throughput | Cache Hit Rate |
|---|---:|---:|---:|
| Baseline (no cache) | 64.1 ms | 15.6 QPS | 0% |
| With MD5 cache | 35.6 ms | 28.1 QPS | 30% |
| **Improvement** | **-44.5%** | **+80.3%** | |

Cache hits averaged **12ms** (memory lookup only). Cache misses averaged **65ms** (embed + HNSW). With 30% hit rate, the blended average dropped from 64ms to 36ms.

At higher repeat rates (a busy Monday morning after a system outage when everyone asks the same thing), the improvement would be significantly larger.

---

## Production Monitoring

Every query is logged to `logs/rag.jsonl` as a structured JSON line:

```json
{
  "timestamp": "2026-05-19T20:31:15.112Z",
  "service": "rag",
  "event": "query",
  "latency_ms": 58.3,
  "query_length": 28,
  "n_results": 3,
  "cache_hit": false,
  "status": "ok",
  "error": null,
  "estimated_cost_usd": 7e-10
}
```

The monitoring dashboard (`week9_10_monitoring/dashboard.py`) reads these files and computes:
- **p50/p95/p99 latency** — average alone is misleading; tail latency is what users feel
- **Error rate** — any ChromaDB connection failures or embedding errors
- **Cache hit rate** — low rate means traffic is non-repetitive or cache is too small
- **QPS and estimated cost** — cost tracking per query

In a 24-hour simulation, the RAG service showed:
- Avg latency: 49.9ms
- p99 latency: 108.5ms (well under the 1000ms CRITICAL threshold)
- Error rate: 3.0%
- Cache hit rate: 28.5%

No alerts fired for the RAG service during the simulation. The chatbot service (with the injected latency spike) was the one that triggered CRITICAL alerts — demonstrating the monitoring system works.

---

## Failure Modes to Watch

**Silent wrong results** — the HNSW index returns the top-K nearest vectors, but "nearest" doesn't mean "correct." If the query is completely out-of-domain (e.g., "who won the 2024 World Cup?"), the system returns the closest FAQ entry anyway, with no indication it's a poor match. Mitigation: filter by distance threshold — if the best match has cosine distance > 0.8, return "I don't know" instead of a confident wrong answer.

**Embedding model drift** — if you rebuild the index with a different model version, old embeddings become incompatible. Always rebuild the full index when changing embedding models. Track model version in ChromaDB collection metadata.

**Cache poisoning** — the MD5 key is based on query text. If query text is sanitized differently in different code paths, two identical queries could generate different keys and miss the cache. Always apply the same normalization (strip + lowercase) before hashing.

---

## Code

Full implementation: [github.com/MadHacker3712/rag-backend](https://github.com/MadHacker3712/rag-backend)  
Monitoring dashboard: [github.com/MadHacker3712/mlops-monitoring](https://github.com/MadHacker3712/mlops-monitoring)

Run it yourself:
```bash
git clone https://github.com/MadHacker3712/rag-backend
cd rag-backend
pip install -r requirements.txt
python build_index.py   # embed 30 FAQ docs into ChromaDB
python benchmark.py     # run baseline vs cached benchmark
python -m uvicorn app:app --port 8000
```
