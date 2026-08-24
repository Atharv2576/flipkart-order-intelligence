"""Embeds every product's natural-language sentence and builds a FAISS
IndexFlatIP over the normalised vectors, mirroring part3/build_index.py's
policy index exactly -- but into data/product_index/, a separate directory,
so the required Part 3 policy index (data/policy_index/) is never touched
by this optional catalog layer.

Run as: python3 -m part3.build_product_index
"""
import json
from pathlib import Path

import faiss

from part3.embeddings import embed
from part3.product_catalog import load_products, product_to_sentence

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "product_index"
INDEX_PATH = INDEX_DIR / "product.faiss"
CHUNKS_PATH = INDEX_DIR / "product_chunks.json"


def main():
    products = load_products()
    chunks = [
        {
            "chunk_id": f"{p['product_id']}::s00",
            "document_id": p["product_id"],
            "document_title": p["product_name"],
            "chunk_index": 0,
            "chunk_text": product_to_sentence(p),
        }
        for p in products
    ]
    texts = [c["chunk_text"] for c in chunks]

    vectors = embed(texts)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    print(f"products: {len(products)}")
    print(f"embedding dim: {vectors.shape[1]}")
    print(f"index written to {INDEX_PATH}")
    return {"products": len(products)}


if __name__ == "__main__":
    main()
