"""Reindex ChromaDB with SentenceTransformer embeddings.

Reads all records from existing collection, re-embeds with sentence-transformers,
and writes to a fresh collection. Run with:
    PYTHONPATH=. python3 scripts/reindex_embeddings.py
"""

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import sys

COLLECTION_NAME = "contextengine"
BATCH_SIZE = 256


def main():
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded: all-MiniLM-L6-v2 (dim={dim})")

    client = chromadb.PersistentClient(path=".chroma_data")

    # Read all records from old collection
    old_col = client.get_collection(COLLECTION_NAME)
    total = old_col.count()
    print(f"Reading {total} records from '{COLLECTION_NAME}'...")

    # Fetch all data
    all_data = old_col.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]

    print(f"Read {len(ids)} records. Re-embedding in batches of {BATCH_SIZE}...")

    # Delete old collection and create fresh
    client.delete_collection(COLLECTION_NAME)
    new_col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Re-embed and upsert in batches
    t0 = time.time()
    for i in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_metas = metadatas[i : i + BATCH_SIZE]

        # Encode
        embeddings = model.encode(batch_docs, normalize_embeddings=True, show_progress_bar=False)

        new_col.upsert(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            documents=batch_docs,
            metadatas=batch_metas,
        )

        done = min(i + BATCH_SIZE, len(ids))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(ids) - done) / rate if rate > 0 else 0
        print(f"  {done}/{len(ids)} ({done*100/len(ids):.1f}%) - {rate:.0f} rec/s - ETA {eta:.0f}s")

    total_time = time.time() - t0
    print(f"\nDone! {len(ids)} records re-indexed in {total_time:.1f}s")
    print(f"New collection count: {new_col.count()}")


if __name__ == "__main__":
    main()
