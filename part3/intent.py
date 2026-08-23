"""Intent classification by nearest few-shot exemplar in embedding space --
the exemplars *are* the classifier, not a decoration next to a keyword
matcher. Editing this list changes routing directly.

The return_risk exemplars are deliberately built around "a specific order
being scored" rather than the bare word "return", because a genuine policy
question like "Can I return a used lipstick?" is lexically closer to
"is this likely to be returned" than to any other policy exemplar if the
return_risk examples are phrased around that shared word. Distinguishing on
what actually marks the intent -- an order id being checked -- keeps the two
lanes separated; see reports/part3_intent_routing.md for the measured check.
"""
import re

from part3.config import INTENT_ROUTING_FLOOR
from part3.embeddings import embed

FEW_SHOT_EXAMPLES = [
    ("How many days do I have to return a mobile phone?", "policy"),
    ("What is the refund timeline for a cash on delivery order?", "policy"),
    ("Can I exchange these shoes for a different size?", "policy"),
    ("How long does delivery usually take to a non-metro city?", "policy"),
    ("Do I need to ship the item back myself, or will someone pick it up?", "policy"),
    ("Can I return a beauty product I already opened?", "policy"),
    ("Score the return risk for order 4521.", "return_risk"),
    ("What is the probability that order 1090 gets returned?", "return_risk"),
    ("Check the return risk on order 3312 for me.", "return_risk"),
    ("How risky is order 782 in terms of a potential return?", "return_risk"),
    ("Run a risk check on order number 6001.", "return_risk"),
    ("What category does this product photo belong to?", "product_category"),
    ("Can you tell me what item is shown in this image?", "product_category"),
    ("Classify this product picture for me.", "product_category"),
    ("What kind of item is in this uploaded photo?", "product_category"),
    ("Identify the catalogue category of this image.", "product_category"),
]

ORDER_ID_PATTERN = re.compile(r"\border\s*(?:id\s*)?#?\s*(\d+)\b", re.IGNORECASE)

_exemplar_vectors = None


def _get_exemplar_vectors():
    global _exemplar_vectors
    if _exemplar_vectors is None:
        texts = [text for text, _ in FEW_SHOT_EXAMPLES]
        _exemplar_vectors = embed(texts)
    return _exemplar_vectors


def classify_intent(message: str) -> dict:
    vectors = _get_exemplar_vectors()
    query_vector = embed([message])[0]

    similarities = vectors @ query_vector  # both L2-normalised -> cosine similarity
    ranked = sorted(
        zip(similarities.tolist(), FEW_SHOT_EXAMPLES), key=lambda pair: pair[0], reverse=True
    )
    best_score, (best_text, best_intent) = ranked[0]

    below_floor = best_score < INTENT_ROUTING_FLOOR
    final_intent = "policy" if below_floor else best_intent

    return {
        "final_intent": final_intent,
        "nearest_example": best_text,
        "nearest_example_intent": best_intent,
        "similarity": round(float(best_score), 4),
        "below_floor": below_floor,
        "runner_up": [
            {"text": text, "intent": intent, "similarity": round(float(score), 4)}
            for score, (text, intent) in ranked[1:3]
        ],
    }


def extract_order_id(message: str) -> int | None:
    match = ORDER_ID_PATTERN.search(message)
    return int(match.group(1)) if match else None
