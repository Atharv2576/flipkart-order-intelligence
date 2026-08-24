"""Loads the synthetic product catalog (data/product_catalog.json, built by
scripts/generate_product_catalog.py) and offers two ways to query it:

- query_products(**filters): a real structured filter over the JSON, for
  questions with an enumerable answer ("which products support COD?",
  "which have a 10-day return window?") -- these should never be answered
  by a semantic-search guess when a direct filter gives an exact answer.
- search_products(query, top_k): semantic search over a small FAISS index
  built by part3/build_product_index.py, for fuzzy product mentions
  ("the Nike-style running shoe") that don't name an exact SKU.
"""
import json
import re
from pathlib import Path

import faiss

from part3.embeddings import embed
from part3.slot_extraction import extract_category

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "product_catalog.json"
INDEX_PATH = ROOT / "data" / "product_index" / "product.faiss"
CHUNKS_PATH = ROOT / "data" / "product_index" / "product_chunks.json"

# Catalog data is synthetic per-SKU filler, not the graded policy KB -- it
# must never outrank or stand in for a genuine policy answer about the
# user's own order. This gate (the same nearest-few-shot-exemplar mechanism
# part3/intent.py uses) only lets catalog evidence into the pool for a
# message that's actually asking about product/catalog info in general
# ("are headphones returnable", "which products support COD"), not one
# describing the user's own purchase ("I bought running shoes, COD, they
# arrived damaged") where a specific random SKU would be actively
# misleading if it out-scored the real policy document.
CATALOG_RELEVANCE_EXAMPLES = [
    "Are headphones returnable?",
    "Which products support Cash on Delivery?",
    "Which products have a 10-day return window?",
    "What's the delivery SLA for this product?",
    "Can I exchange this product instead of returning it?",
    "Which products are non-returnable?",
    "Tell me about this product's warranty.",
    "Does this item come with a warranty?",
]
CATALOG_RELEVANCE_FLOOR = 0.45

_products = None
_index = None
_chunks = None
_relevance_vectors = None


def _get_relevance_vectors():
    global _relevance_vectors
    if _relevance_vectors is None:
        _relevance_vectors = embed(CATALOG_RELEVANCE_EXAMPLES)
    return _relevance_vectors


def is_catalog_relevant(text: str) -> bool:
    vectors = _get_relevance_vectors()
    query_vector = embed([text])[0]
    best_score = float((vectors @ query_vector).max())
    return best_score >= CATALOG_RELEVANCE_FLOOR


def load_products() -> list[dict]:
    global _products
    if _products is None:
        _products = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return _products


def product_to_sentence(product: dict) -> str:
    """The natural-language sentence indexed for this product -- the same
    sentence both FAISS search and a human would read as the source.
    """
    if product["non_returnable"]:
        return_clause = "is not eligible for return"
    else:
        return_clause = f"can be returned within {product['return_window']} days of delivery"

    exchange_clause = "supports exchange" if product["exchange_available"] else "does not support exchange"
    cod_clause = "is available for Cash on Delivery" if product["cod_available"] else "is prepaid-only, not available for Cash on Delivery"
    warranty_clause = (
        f"carries a {product['warranty']} warranty" if product["warranty"] != "No warranty" else "carries no warranty"
    )

    return (
        f"The {product['product_name']} ({product['category']} > {product['subcategory']}, "
        f"₹{product['price_inr']:.0f}) {return_clause}, {exchange_clause}, {cod_clause}, "
        f"and {warranty_clause}."
    )


def query_products(
    category: str | None = None,
    subcategory: str | None = None,
    cod_available: bool | None = None,
    exchange_available: bool | None = None,
    non_returnable: bool | None = None,
    min_return_window: int | None = None,
    max_return_window: int | None = None,
) -> list[dict]:
    results = load_products()
    if category is not None:
        results = [p for p in results if p["category"].lower() == category.lower()]
    if subcategory is not None:
        results = [p for p in results if p["subcategory"].lower() == subcategory.lower()]
    if cod_available is not None:
        results = [p for p in results if p["cod_available"] == cod_available]
    if exchange_available is not None:
        results = [p for p in results if p["exchange_available"] == exchange_available]
    if non_returnable is not None:
        results = [p for p in results if p["non_returnable"] == non_returnable]
    if min_return_window is not None:
        results = [p for p in results if (p["return_window"] or 0) >= min_return_window]
    if max_return_window is not None:
        results = [
            p for p in results if p["return_window"] is not None and p["return_window"] <= max_return_window
        ]
    return results


_STRUCTURED_MARKER = re.compile(r"\bwhich\b|\bwhat\b|\blist\b", re.IGNORECASE)
_RETURN_WINDOW_PATTERN = re.compile(r"(\d+)[\s-]*day", re.IGNORECASE)
_COD_PATTERN = re.compile(r"\bcod\b|cash on delivery", re.IGNORECASE)
_NON_RETURNABLE_PATTERN = re.compile(r"non-returnable|not returnable|cannot be returned|can't be returned", re.IGNORECASE)
_EXCHANGE_PATTERN = re.compile(r"\bexchange\b", re.IGNORECASE)


def is_structured_filter_query(text: str) -> bool:
    """A "which/what products ..." question has an exact, enumerable answer
    -- it should be answered with a real filter over the catalog, never a
    semantic-search guess. Requires both the generic English question
    structure ("which"/"what"/"list") and the word "product(s)", so this
    doesn't fire on an ordinary "what category is this image" question.
    """
    lower = text.lower()
    return bool(_STRUCTURED_MARKER.search(lower)) and "product" in lower


def parse_filter_query(text: str) -> dict:
    """Extracts structured filter criteria from a "which products ..."
    question. Matches domain values (COD/non-returnable/exchange/category)
    against the catalog's own closed vocabulary -- this populates query
    parameters for a real database-style filter, not a canned answer.
    """
    filters: dict = {}
    lower = text.lower()

    if _COD_PATTERN.search(lower):
        filters["cod_available"] = True
    if _NON_RETURNABLE_PATTERN.search(lower):
        filters["non_returnable"] = True
    elif _EXCHANGE_PATTERN.search(lower):
        filters["exchange_available"] = True

    window_match = _RETURN_WINDOW_PATTERN.search(lower)
    if window_match:
        days = int(window_match.group(1))
        if "at least" in lower or "or more" in lower or "minimum" in lower:
            filters["min_return_window"] = days
        elif "at most" in lower or "or less" in lower or "within" in lower or "up to" in lower:
            filters["max_return_window"] = days
        else:
            filters["min_return_window"] = days
            filters["max_return_window"] = days

    category = extract_category(text)
    if category:
        filters["category"] = category

    return filters


def compose_structured_filter_answer(filters: dict, results: list[dict]) -> dict:
    if not results:
        answer = "No products in the catalog match that filter."
    else:
        names = [f"{p['product_name']} ({p['category']})" for p in results[:8]]
        remainder = len(results) - len(names)
        more = f", and {remainder} more" if remainder > 0 else ""
        answer = f"{len(results)} product(s) in the catalog match: " + "; ".join(names) + more + "."
    return {"answer": answer, "source": "policy_kb", "confidence": 1.0}


def _load_index():
    global _index, _chunks
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
        _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return _index, _chunks


def search_products(query: str, top_k: int = 3) -> list[dict]:
    """Semantic search over product sentences, returning hits shaped like
    part3/retrieval.py's chunk hits (document_id/document_title/chunk_text/
    score) plus the full product record, so callers can merge product and
    policy evidence in one pool.
    """
    index, chunks = _load_index()
    query_vector = embed([query])
    scores, positions = index.search(query_vector, top_k)

    products_by_id = {p["product_id"]: p for p in load_products()}
    results = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        hit = dict(chunks[position])
        hit["score"] = float(score)
        hit["evidence_type"] = "product"
        hit["product"] = products_by_id.get(hit["document_id"])
        results.append(hit)
    return results
