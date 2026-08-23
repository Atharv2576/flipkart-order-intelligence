"""Query-time retrieval: embed the query, search the FAISS index, and roll
chunk hits up to unique parent documents (retrieval evaluation and the
groundedness check both operate at the document level).
"""
import json
from pathlib import Path

import faiss

from part3.embeddings import embed

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data" / "policy_index" / "policy.faiss"
CHUNKS_PATH = ROOT / "data" / "policy_index" / "chunks.json"

DEFAULT_TOP_K = 3

_index = None
_chunks = None


def _load():
    global _index, _chunks
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
        _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return _index, _chunks


def search_chunks(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    index, chunks = _load()
    query_vector = embed([query])
    scores, positions = index.search(query_vector, top_k)

    results = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        hit = dict(chunks[position])
        hit["score"] = float(score)
        results.append(hit)
    return results


def rollup_to_documents(chunk_hits: list[dict]) -> list[dict]:
    """Deduplicate chunk hits to their unique parent documents, keeping each
    document's single best-scoring chunk.
    """
    best_by_doc: dict[str, dict] = {}
    for hit in chunk_hits:
        doc_id = hit["document_id"]
        if doc_id not in best_by_doc or hit["score"] > best_by_doc[doc_id]["score"]:
            best_by_doc[doc_id] = hit
    return sorted(best_by_doc.values(), key=lambda h: h["score"], reverse=True)


def search_documents(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    return rollup_to_documents(search_chunks(query, top_k))


def check_groundedness(chunk_hits: list[dict], threshold: float) -> dict:
    if not chunk_hits:
        return {"grounded": False, "best_score": 0.0, "threshold": threshold}
    best_score = max(hit["score"] for hit in chunk_hits)
    return {
        "grounded": best_score >= threshold,
        "best_score": round(best_score, 4),
        "threshold": threshold,
    }
