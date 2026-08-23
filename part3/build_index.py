"""Embeds every policy chunk and builds a FAISS IndexFlatIP over the
normalised vectors (inner product on L2-normalised vectors == cosine
similarity). Persists the index and chunk metadata to data/policy_index/.

Run as: python3 -m part3.build_index
"""
import json
from pathlib import Path

import faiss

from part3.chunking import chunk_documents
from part3.embeddings import embed
from part3.kb import load_policy_documents

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "policy_index"
INDEX_PATH = INDEX_DIR / "policy.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


def main():
    documents = load_policy_documents()
    chunks = chunk_documents(documents)
    texts = [c["chunk_text"] for c in chunks]

    vectors = embed(texts)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    print(f"documents: {len(documents)}")
    print(f"chunks: {len(chunks)}")
    print(f"embedding dim: {vectors.shape[1]}")
    print(f"index written to {INDEX_PATH}")
    return {"documents": len(documents), "chunks": len(chunks)}


if __name__ == "__main__":
    main()
