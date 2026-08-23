"""The default, required response-generation mode: a deterministic
rule/template function that composes the final answer from retrieved KB
chunks or tool output only. Zero network calls, zero API keys, fully
reproducible. Because it can only quote what it was handed, it cannot
fabricate a policy or a number.
"""


def compose_policy_answer(documents: list[dict], groundedness: dict) -> dict:
    if not groundedness["grounded"]:
        return {
            "answer": (
                "I don't have a policy in my knowledge base that covers this with enough "
                f"confidence to answer (best match scored {groundedness['best_score']}, "
                f"below the {groundedness['threshold']} groundedness threshold). "
                "I'd rather say that than guess."
            ),
            "source": "policy_kb",
            "confidence": groundedness["best_score"],
        }

    primary = documents[0]
    answer = primary["chunk_text"]
    answer += f" [Source: {primary['document_title']} ({primary['document_id']})]"

    related = [d for d in documents[1:] if d["document_id"] != primary["document_id"]][:2]
    if related:
        related_str = ", ".join(f"{d['document_title']} ({d['document_id']})" for d in related)
        answer += f" [Related policies: {related_str}]"

    return {"answer": answer, "source": "policy_kb", "confidence": groundedness["best_score"]}


def compose_return_risk_answer(order_id: int | None, tool_result: dict) -> dict:
    probability = tool_result["return_probability"]
    bucket = tool_result["risk_bucket"]
    confidence = round(max(probability, 1 - probability), 4)

    subject = f"Order {order_id}" if order_id is not None else "This order"
    answer = (
        f"{subject} has a {probability * 100:.2f}% predicted probability of being "
        f"returned, which is a {bucket.upper()} risk (threshold t*_rf = "
        f"{tool_result['threshold_rf']})."
    )
    return {"answer": answer, "source": "return_risk_tool", "confidence": confidence}


def compose_product_category_answer(tool_result: dict) -> dict:
    answer = (
        f"This looks like a {tool_result['predicted_class']} "
        f"(confidence {tool_result['confidence'] * 100:.2f}%)."
    )
    return {"answer": answer, "source": "image_classifier_tool", "confidence": tool_result["confidence"]}


def compose_missing_input_answer(kind: str) -> dict:
    if kind == "order_id":
        answer = (
            "I need an order id to check return risk -- mention one like "
            "\"order 1234\" and I can look it up."
        )
        source = "return_risk_tool"
    else:
        answer = "I need an image path to classify a product photo."
        source = "image_classifier_tool"
    return {"answer": answer, "source": source, "confidence": 0.0}


def compose_blocked_answer(matched_patterns: list[str]) -> dict:
    return {
        "answer": (
            "I can't follow instructions embedded in your message that try to override "
            "how I'm supposed to behave. I'm still happy to help with an order, return-risk, "
            "or policy question asked directly."
        ),
        "source": "policy_kb",
        "confidence": 1.0,
    }


def compose_out_of_scope_answer() -> dict:
    return {
        "answer": (
            "That's outside what I can help with -- I can answer Flipkart policy questions, "
            "check an order's return risk, or classify a product photo."
        ),
        "source": "policy_kb",
        "confidence": 0.0,
    }
