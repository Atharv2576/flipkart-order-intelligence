"""Query-time retrieval: embed the query, search the FAISS index, and roll
chunk hits up to unique parent documents (retrieval evaluation and the
groundedness check both operate at the document level).
"""
import json
from pathlib import Path

import faiss

from part3.config import MULTI_CLAUSE_CONFIDENCE_FLOOR
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


def search_chunks_with_catalog(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """search_chunks, plus the same query run against the product catalog's
    own FAISS index (part3/product_catalog.py) -- but only when the query
    itself is asking about product/catalog info (is_catalog_relevant), so a
    message describing the user's own order never gets a random synthetic
    SKU standing in for the real policy answer.
    """
    from part3.product_catalog import is_catalog_relevant, search_products

    hits = search_chunks(query, top_k)
    if is_catalog_relevant(query):
        hits.extend(search_products(query, top_k=2))
    return sorted(hits, key=lambda h: h["score"], reverse=True)


def retrieve_for_message(message: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Clause-aware retrieval. Splits the message into candidate clauses
    (part3/query_decomposition.split_clauses); if it doesn't structurally
    decompose, or the clauses don't each ground on a *different* document,
    this degrades to exactly today's single-query behaviour. Only a genuine
    multi-topic message (e.g. "compare footwear and electronics returns",
    or "...damaged, and I paid COD, and it arrived late") produces
    multi-clause evidence.
    """
    from part3.query_decomposition import split_clauses

    clauses = split_clauses(message)

    if len(clauses) < 2:
        hits = search_chunks_with_catalog(message, top_k)
        return {"mode": "single", "chunk_hits": hits, "clause_evidence": []}

    clause_evidence = []
    seen_doc_ids = set()
    for clause in clauses:
        hits = search_chunks_with_catalog(clause, top_k=top_k)
        documents = rollup_to_documents(hits)
        clause_evidence.append({"clause": clause, "chunk_hits": hits, "documents": documents})
        if documents and documents[0]["score"] >= MULTI_CLAUSE_CONFIDENCE_FLOOR:
            seen_doc_ids.add(documents[0]["document_id"])

    if len(seen_doc_ids) < 2:
        # The clauses didn't actually diverge onto different evidence --
        # not a real comparison, fall back to a single whole-message search.
        hits = search_chunks_with_catalog(message, top_k)
        return {"mode": "single", "chunk_hits": hits, "clause_evidence": []}

    all_hits = [hit for ce in clause_evidence for hit in ce["chunk_hits"]]
    return {"mode": "multi", "chunk_hits": all_hits, "clause_evidence": clause_evidence}
