"""
Week 7-8: RAG Benchmark — Baseline vs Query-Cached

Compares two retrieval setups across 20 test queries:
  - Baseline: embed → ChromaDB search → return top doc (no cache)
  - Cached:   check dict cache first → on miss, embed → search → cache result

Metrics measured:
  - Embedding latency (ms)   — how long to convert query to vector
  - Retrieval latency (ms)   — how long ChromaDB takes to search
  - Total latency (ms)       — end-to-end per query
  - Cache hit rate (%)       — fraction of queries served from cache
"""
import hashlib
import json
import time
from pathlib import Path

import psutil
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

CHROMA_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "customer_support_faq"
EMBED_MODEL = "all-MiniLM-L6-v2"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TOP_K = 3

# 20 test queries — 6 are repeats to simulate real-world cache hits
TEST_QUERIES = [
    "How do I reset my password?",
    "Where is my order?",
    "What is your return policy?",
    "How do I cancel my subscription?",
    "When will my package arrive?",
    "How do I update my payment method?",
    "I received a damaged product",
    "How do I track my shipment?",
    "How do I reset my password?",       # repeat
    "How do I cancel my order?",
    "What is your return policy?",       # repeat
    "How do I get a refund?",
    "My account is locked",
    "How do I change my delivery address?",
    "Where is my order?",                # repeat
    "How do I cancel my subscription?",  # repeat
    "What payment methods do you accept?",
    "How long does shipping take?",
    "How do I reset my password?",       # repeat
    "I received a damaged product",      # repeat
]


def get_rss_mb():
    return psutil.Process().memory_info().rss / (1024 ** 2)


def load_resources():
    print(f"Loading embedding model ({EMBED_MODEL})...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    print(f"  Collection has {collection.count()} documents\n")
    return embed_model, collection


# ── Setup 1: Baseline (no cache) ─────────────────────────────────────────────
def benchmark_baseline(embed_model, collection, queries):
    results = []
    mem_before = get_rss_mb()

    for query in queries:
        # Step 1: embed
        t0 = time.perf_counter()
        embedding = embed_model.encode([query])
        embed_ms = (time.perf_counter() - t0) * 1000

        # Step 2: search ChromaDB
        t1 = time.perf_counter()
        docs = collection.query(
            query_embeddings=embedding.tolist(),
            n_results=TOP_K,
            include=["metadatas", "distances"],
        )
        retrieve_ms = (time.perf_counter() - t1) * 1000

        top_answer = docs["metadatas"][0][0]["answer"]
        total_ms = embed_ms + retrieve_ms

        results.append({
            "query": query,
            "embed_ms": round(embed_ms, 2),
            "retrieve_ms": round(retrieve_ms, 2),
            "total_ms": round(total_ms, 2),
            "top_answer_preview": top_answer[:60] + "...",
        })

    mem_after = get_rss_mb()
    avg_embed = sum(r["embed_ms"] for r in results) / len(results)
    avg_retrieve = sum(r["retrieve_ms"] for r in results) / len(results)
    avg_total = sum(r["total_ms"] for r in results) / len(results)

    return {
        "setup": "Baseline (no cache)",
        "avg_embed_ms": round(avg_embed, 2),
        "avg_retrieve_ms": round(avg_retrieve, 2),
        "avg_total_ms": round(avg_total, 2),
        "throughput_qps": round(1000 / avg_total, 3),
        "peak_memory_mb": round(mem_after, 1),
        "cache_hit_rate_pct": 0.0,
        "per_query": results,
    }


# ── Setup 2: Query Cache ──────────────────────────────────────────────────────
def benchmark_cached(embed_model, collection, queries):
    cache: dict = {}
    results = []
    hits = 0
    mem_before = get_rss_mb()

    for query in queries:
        cache_key = hashlib.md5(query.strip().lower().encode()).hexdigest()

        t_start = time.perf_counter()

        if cache_key in cache:
            # Cache hit — return immediately, no embedding or DB call needed
            hits += 1
            total_ms = (time.perf_counter() - t_start) * 1000
            results.append({
                "query": query,
                "cache_hit": True,
                "embed_ms": 0.0,
                "retrieve_ms": 0.0,
                "total_ms": round(total_ms, 2),
            })
        else:
            # Cache miss — full retrieval pipeline
            t0 = time.perf_counter()
            embedding = embed_model.encode([query])
            embed_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            docs = collection.query(
                query_embeddings=embedding.tolist(),
                n_results=TOP_K,
                include=["metadatas", "distances"],
            )
            retrieve_ms = (time.perf_counter() - t1) * 1000

            top_answer = docs["metadatas"][0][0]["answer"]
            cache[cache_key] = top_answer
            total_ms = embed_ms + retrieve_ms

            results.append({
                "query": query,
                "cache_hit": False,
                "embed_ms": round(embed_ms, 2),
                "retrieve_ms": round(retrieve_ms, 2),
                "total_ms": round(total_ms, 2),
            })

    mem_after = get_rss_mb()
    avg_embed = sum(r["embed_ms"] for r in results) / len(results)
    avg_retrieve = sum(r["retrieve_ms"] for r in results) / len(results)
    avg_total = sum(r["total_ms"] for r in results) / len(results)
    hit_rate = (hits / len(queries)) * 100

    return {
        "setup": "Cached (MD5 exact-match cache)",
        "avg_embed_ms": round(avg_embed, 2),
        "avg_retrieve_ms": round(avg_retrieve, 2),
        "avg_total_ms": round(avg_total, 2),
        "throughput_qps": round(1000 / avg_total, 3),
        "peak_memory_mb": round(mem_after, 1),
        "cache_hit_rate_pct": round(hit_rate, 1),
        "per_query": results,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_table(r1, r2):
    print("\n" + "=" * 80)
    print(f"{'Setup':<32} {'Embed':>8} {'Retrieve':>10} {'Total':>8} {'QPS':>8} {'HitRate':>9}")
    print(f"{'':32} {'(ms)':>8} {'(ms)':>10} {'(ms)':>8} {'':>8} {'(%)':>9}")
    print("=" * 80)
    for r in [r1, r2]:
        print(
            f"{r['setup']:<32}"
            f" {r['avg_embed_ms']:>8.1f}"
            f" {r['avg_retrieve_ms']:>10.1f}"
            f" {r['avg_total_ms']:>8.1f}"
            f" {r['throughput_qps']:>8.3f}"
            f" {r['cache_hit_rate_pct']:>9.1f}"
        )
    print("=" * 80)

    lat_imp = (r1["avg_total_ms"] - r2["avg_total_ms"]) / r1["avg_total_ms"] * 100
    tput_imp = (r2["throughput_qps"] - r1["throughput_qps"]) / r1["throughput_qps"] * 100
    print(f"\nTotal latency improvement (cached vs baseline): {lat_imp:+.1f}%")
    print(f"Throughput improvement   (cached vs baseline): {tput_imp:+.1f}%")
    print(f"Cache hit rate: {r2['cache_hit_rate_pct']}%  ({int(r2['cache_hit_rate_pct'] * len(TEST_QUERIES) / 100)} / {len(TEST_QUERIES)} queries served from cache)")


def main():
    embed_model, collection = load_resources()

    print(f"Benchmarking {len(TEST_QUERIES)} queries...\n")

    print("▶ Setup 1: Baseline (no cache)...")
    r1 = benchmark_baseline(embed_model, collection, TEST_QUERIES)
    print(f"  avg total: {r1['avg_total_ms']:.1f} ms")

    print("▶ Setup 2: Cached (exact-match MD5 cache)...")
    r2 = benchmark_cached(embed_model, collection, TEST_QUERIES)
    print(f"  avg total: {r2['avg_total_ms']:.1f} ms  |  hit rate: {r2['cache_hit_rate_pct']}%")

    print_table(r1, r2)

    out = {
        "baseline": {k: v for k, v in r1.items() if k != "per_query"},
        "cached": {k: v for k, v in r2.items() if k != "per_query"},
        "num_queries": len(TEST_QUERIES),
        "embed_model": EMBED_MODEL,
        "top_k": TOP_K,
    }
    out_path = RESULTS_DIR / "rag_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
