"""
Build ChromaDB vector index from FAQ data.

Run this once before using the RAG pipeline:
    python build_index.py

Creates a persistent ChromaDB index at ./chroma_db/
Uses sentence-transformers all-MiniLM-L6-v2 for embeddings (22MB, fast).
"""
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

FAQ_PATH = Path(__file__).parent / "data" / "faq.json"
CHROMA_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "customer_support_faq"
EMBED_MODEL = "all-MiniLM-L6-v2"


def build_index():
    print(f"Loading FAQ data from {FAQ_PATH}...")
    with open(FAQ_PATH) as f:
        faq = json.load(f)
    print(f"  {len(faq)} FAQ entries loaded")

    print(f"\nInitializing ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection")
    except Exception:
        pass

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Index questions as the searchable text; store full Q+A as metadata
    print(f"\nEmbedding {len(faq)} documents with {EMBED_MODEL}...")
    collection.add(
        ids=[entry["id"] for entry in faq],
        documents=[entry["question"] for entry in faq],
        metadatas=[
            {
                "category": entry["category"],
                "question": entry["question"],
                "answer": entry["answer"],
            }
            for entry in faq
        ],
    )

    count = collection.count()
    print(f"  Index built: {count} documents stored")
    print(f"\nChromaDB index saved to: {CHROMA_DIR}")
    print("Ready to run benchmark.py or app.py")


if __name__ == "__main__":
    build_index()
