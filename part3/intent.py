"""Intent classification by nearest few-shot exemplar in embedding space --
the exemplars *are* the classifier, not a decoration next to a keyword
matcher. Editing this list changes routing directly.

The return_risk exemplars are deliberately built around "a specific order
being scored" rather than the bare word "return", because a genuine policy
question like "Can I return a used lipstick?" is lexically closer to
"is this likely to be returned" than to any other policy exemplar if the
return_risk examples are phrased around that shared word. Distinguishing on
what actually marks the intent -- an order id being checked -- keeps the two
lanes separated; see reports/part3_retrieval_evaluation.md for related
measurement notes.

`conversational` is a fifth exemplar bucket (greeting / general-help /
thanks / farewell) sitting alongside the three required intents. It exists
because, without it, a plain "hi" has no close exemplar at all -- it lands
near a policy example at very low similarity, falls below the routing
floor, and used to be silently absorbed into `policy`, where retrieval then
finds nothing and returns a groundedness refusal that reads exactly like a
denial of service. Two independent floors (see part3/config.py) separate
three outcomes: a genuine intent match, a genuine conversational match, or
neither (`unsupported`) -- rather than only ever falling back to `policy`.
"""
import re

from part3.config import CONVERSATIONAL_ROUTING_FLOOR, INTENT_ROUTING_FLOOR
from part3.embeddings import embed

FEW_SHOT_EXAMPLES = [
    ("How many days do I have to return a mobile phone?", "policy"),
    ("What is the refund timeline for a cash on delivery order?", "policy"),
    ("Can I exchange these shoes for a different size?", "policy"),
    ("How long does delivery usually take to a non-metro city?", "policy"),
    ("Do I need to ship the item back myself, or will someone pick it up?", "policy"),
    ("Can I return a beauty product I already opened?", "policy"),
    ("Can you explain the return policy for footwear in simple terms?", "policy"),
    ("Summarize the return policies that apply to my order.", "policy"),
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
    ("Hi", "conversational"),
    ("Hello there", "conversational"),
    ("Hey, is anyone there?", "conversational"),
    ("Good morning", "conversational"),
    ("What can you help me with?", "conversational"),
    ("What can you do?", "conversational"),
    ("How does this assistant work?", "conversational"),
    ("Tell me about yourself.", "conversational"),
    ("Thanks, that helps.", "conversational"),
    ("Thank you very much!", "conversational"),
    ("Okay, bye for now.", "conversational"),
    ("That's all I needed, goodbye.", "conversational"),
]

ORDER_ID_PATTERN = re.compile(r"\border\s*(?:id\s*)?#?\s*(\d+)\b", re.IGNORECASE)

CONVERSATIONAL_SUBTYPES = {
    "Hi": "greeting",
    "Hello there": "greeting",
    "Hey, is anyone there?": "greeting",
    "Good morning": "greeting",
    "What can you help me with?": "general_help",
    "What can you do?": "general_help",
    "How does this assistant work?": "general_help",
    "Tell me about yourself.": "general_help",
    "Thanks, that helps.": "thanks",
    "Thank you very much!": "thanks",
    "Okay, bye for now.": "farewell",
    "That's all I needed, goodbye.": "farewell",
}

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

    # Required-intent and conversational exemplars are ranked separately so
    # that adding the conversational bucket cannot change the required-floor
    # fallback behaviour for a message that's neither -- the below_floor
    # check below is computed exactly as it was before conversational
    # exemplars existed.
    required_ranked = [pair for pair in ranked if pair[1][1] != "conversational"]
    conversational_ranked = [pair for pair in ranked if pair[1][1] == "conversational"]

    best_score, (best_text, best_intent) = required_ranked[0]
    best_conv_score, (best_conv_text, _) = conversational_ranked[0]

    below_floor = best_score < INTENT_ROUTING_FLOOR
    conversational_wins = (
        best_conv_score >= CONVERSATIONAL_ROUTING_FLOOR and best_conv_score > best_score
    )

    if conversational_wins:
        final_intent = "conversational"
    else:
        final_intent = "policy" if below_floor else best_intent

    return {
        "final_intent": final_intent,
        "nearest_example": best_conv_text if conversational_wins else best_text,
        "nearest_example_intent": "conversational" if conversational_wins else best_intent,
        "similarity": round(float(best_conv_score if conversational_wins else best_score), 4),
        "below_floor": below_floor,
        "conversational_subtype": (
            CONVERSATIONAL_SUBTYPES.get(best_conv_text) if conversational_wins else None
        ),
        "runner_up": [
            {"text": text, "intent": intent, "similarity": round(float(score), 4)}
            for score, (text, intent) in ranked[1:3]
        ],
    }


def extract_order_id(message: str) -> int | None:
    match = ORDER_ID_PATTERN.search(message)
    return int(match.group(1)) if match else None
