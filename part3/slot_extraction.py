"""Free-text slot extraction for the return-risk tool's order_features
input. This is structured form-filling against the model's own closed
vocabulary (part1.common's exact category/payment-method values, taken
verbatim from generate_orders.py), not the banned "if question contains X"
pattern -- it populates data fields, it never selects a canned answer.

product_category is resolved with the same nearest-few-shot-exemplar
mechanism part3/intent.py uses for intent routing, applied to short
candidate word spans rather than the whole message -- embedding an entire
sentence like "I ordered a sneaker using COD, it arrived 6 days late"
dilutes the one word ("sneaker") that actually names the product among
payment/delivery noise, so "sneaker"/"trainers"/"running shoes" all resolve
to Footwear by matching short spans against category exemplars instead.
"""
import re

from part3.embeddings import embed

PAYMENT_METHODS = ["COD", "Prepaid_Card", "Prepaid_UPI", "Wallet"]
CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]

CATEGORY_EXAMPLES = [
    ("a t-shirt", "Apparel"), ("a dress", "Apparel"), ("jeans", "Apparel"),
    ("a jacket", "Apparel"), ("ethnic wear", "Apparel"), ("a shirt", "Apparel"),
    ("a hoodie", "Apparel"), ("clothes", "Apparel"),
    ("running shoes", "Footwear"), ("sneakers", "Footwear"), ("sandals", "Footwear"),
    ("boots", "Footwear"), ("slippers", "Footwear"), ("formal shoes", "Footwear"),
    ("a smartphone", "Electronics"), ("headphones", "Electronics"), ("a laptop", "Electronics"),
    ("earbuds", "Electronics"), ("a smartwatch", "Electronics"), ("a charger", "Electronics"),
    ("a power bank", "Electronics"), ("a monitor", "Electronics"), ("a keyboard", "Electronics"),
    ("a mouse", "Electronics"),
    ("a kitchen appliance", "Home"), ("a lamp", "Home"), ("bedsheets", "Home"),
    ("furniture", "Home"), ("home decor", "Home"), ("a storage unit", "Home"),
    ("skincare", "Beauty"), ("makeup", "Beauty"), ("a fragrance", "Beauty"), ("haircare", "Beauty"),
]

_STOPWORDS = {
    "a", "an", "the", "is", "it", "i", "my", "for", "and", "of", "to", "in",
    "on", "using", "with", "was", "arrived", "ordered", "bought", "paid",
    "cost", "using", "days", "day", "late", "delivery", "order",
}

_PAYMENT_PATTERNS = [
    (re.compile(r"\bcash on delivery\b|\bcod\b", re.IGNORECASE), "COD"),
    (re.compile(r"\bupi\b|\bgpay\b|\bpaytm\b|\bphonepe\b", re.IGNORECASE), "Prepaid_UPI"),
    (re.compile(r"\bwallet\b", re.IGNORECASE), "Wallet"),
    (re.compile(r"\bcredit card\b|\bdebit card\b|\bprepaid card\b|\bcard\b", re.IGNORECASE), "Prepaid_Card"),
]

_PRICE_PATTERN = re.compile(
    r"(?:₹|rs\.?|inr|rupees?)\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*(?:rupees?|rs\.?|inr)",
    re.IGNORECASE,
)

_DELIVERY_DAYS_PATTERN = re.compile(
    r"(\d+)\s*days?\s*(?:late|delayed|overdue)|(?:took|arrived in|delivered in)\s*(\d+)\s*days?",
    re.IGNORECASE,
)

_CATEGORY_MATCH_FLOOR = 0.55

_category_example_vectors = None


def _get_category_example_vectors():
    global _category_example_vectors
    if _category_example_vectors is None:
        _category_example_vectors = embed([text for text, _ in CATEGORY_EXAMPLES])
    return _category_example_vectors


def _candidate_spans(text: str) -> list[str]:
    words = [w for w in re.findall(r"[A-Za-z']+", text.lower()) if w not in _STOPWORDS]
    spans = set()
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            spans.add(" ".join(words[i : i + n]))
    return list(spans)


def _extract_price(text: str) -> float | None:
    match = _PRICE_PATTERN.search(text)
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _extract_payment_method(text: str) -> str | None:
    for pattern, value in _PAYMENT_PATTERNS:
        if pattern.search(text):
            return value
    return None


def _extract_delivery_days(text: str) -> int | None:
    match = _DELIVERY_DAYS_PATTERN.search(text)
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    return int(raw)


def extract_category(text: str) -> str | None:
    spans = _candidate_spans(text)
    if not spans:
        return None

    example_vectors = _get_category_example_vectors()
    span_vectors = embed(spans)
    similarity_matrix = span_vectors @ example_vectors.T  # (n_spans, n_examples)

    best_flat_index = int(similarity_matrix.argmax())
    best_span_index, best_example_index = divmod(best_flat_index, similarity_matrix.shape[1])
    best_score = float(similarity_matrix[best_span_index, best_example_index])

    if best_score < _CATEGORY_MATCH_FLOOR:
        return None
    return CATEGORY_EXAMPLES[best_example_index][1]


def extract_order_features_from_text(text: str) -> dict:
    """Returns a partial dict with only the keys it found evidence for --
    never a full FEATURES dict, so callers must merge this over what's
    already known rather than treating it as a complete order.
    """
    extracted = {}
    price = _extract_price(text)
    if price is not None:
        extracted["price_inr"] = price
    payment_method = _extract_payment_method(text)
    if payment_method is not None:
        extracted["payment_method"] = payment_method
    category = extract_category(text)
    if category is not None:
        extracted["product_category"] = category
    delivery_days = _extract_delivery_days(text)
    if delivery_days is not None:
        extracted["delivery_days"] = delivery_days
    return extracted
