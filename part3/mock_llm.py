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


def compose_return_risk_answer(
    order_id: int | None, tool_result: dict, order_features: dict | None = None
) -> dict:
    probability = tool_result["return_probability"]
    bucket = tool_result["risk_bucket"]
    confidence = round(max(probability, 1 - probability), 4)

    subject = f"Order {order_id}" if order_id is not None else "This order"
    answer = (
        f"{subject} has a {probability * 100:.2f}% predicted probability of being "
        f"returned, which is a {bucket.upper()} risk (threshold t*_rf = "
        f"{tool_result['threshold_rf']})."
    )

    if order_features:
        from part3.risk_explanation import explain_risk

        clauses = explain_risk(order_features)
        if clauses:
            answer += (
                " The strongest contributors, per this model's own held-out "
                "permutation-importance analysis, are " + "; ".join(clauses) + "."
            )

    return {"answer": answer, "source": "return_risk_tool", "confidence": confidence}


def compose_missing_features_answer(missing_fields: list[str], known_features: dict) -> dict:
    """Used when return-risk fires with some order details known (from
    conversation so far) but not enough for a non-degenerate estimate --
    asks only for what's still missing, never for fields already known.
    """
    field_labels = {
        "product_category": "the product category",
        "price_inr": "the price",
        "payment_method": "the payment method (COD, card, UPI, or wallet)",
        "delivery_days": "how many days delivery took",
    }
    missing_text = ", ".join(field_labels.get(f, f) for f in missing_fields)
    known_text = ""
    if known_features:
        known_bits = [f"{k}={v}" for k, v in known_features.items() if v is not None]
        if known_bits:
            known_text = f" I already have: {', '.join(known_bits)}."

    answer = (
        f"I can estimate that. I still need {missing_text}.{known_text} "
        "Mention them and I can run the risk model, or give me an order id and "
        "I'll look the order up directly."
    )
    return {"answer": answer, "source": "return_risk_tool", "confidence": 0.0}


def compose_comparison_answer(clause_evidence: list[dict], threshold: float) -> dict:
    """One grounded statement per clause -- fixes both the "compare X and Y"
    case and a multi-part question (damaged + COD + footwear in one
    message) with the same mechanism: each clause got its own retrieval,
    each clause gets its own answer, instead of one chunk standing in for
    the whole question.
    """
    from part3.retrieval import check_groundedness

    parts = []
    best_score = 0.0
    any_grounded = False

    for evidence in clause_evidence:
        groundedness = check_groundedness(evidence["chunk_hits"], threshold)
        best_score = max(best_score, groundedness["best_score"])
        documents = evidence["documents"]
        if groundedness["grounded"] and documents:
            any_grounded = True
            primary = documents[0]
            parts.append(
                f"On {evidence['clause']}: {primary['chunk_text']} "
                f"[Source: {primary['document_title']} ({primary['document_id']})]"
            )
        else:
            parts.append(
                f"On {evidence['clause']}: I couldn't find a policy covering this with "
                f"enough confidence (best match {groundedness['best_score']}, below the "
                f"{threshold} threshold)."
            )

    answer = " ".join(parts)
    return {
        "answer": answer,
        "source": "policy_kb",
        "confidence": round(best_score, 4) if any_grounded else 0.0,
    }


_CONVERSATIONAL_ANSWERS = {
    "greeting": (
        "Hi! I'm the Flipkart order support assistant. I can help with returns, "
        "refunds, delivery, exchanges, product information, and return-risk scoring "
        "-- ask me anything about an order."
    ),
    "general_help": (
        "I can answer policy questions (returns, refunds, exchanges, delivery, "
        "damaged/wrong items), score an order's return risk if you give me an order id "
        "or a few order details, and classify a product photo if you attach one. "
        "Ask in plain language -- I don't need an exact phrasing."
    ),
    "thanks": "You're welcome -- let me know if there's anything else about an order or policy.",
    "farewell": "Goodbye! Come back any time you have an order, returns, or policy question.",
}


def compose_conversational_answer(subtype: str | None) -> dict:
    answer = _CONVERSATIONAL_ANSWERS.get(subtype, _CONVERSATIONAL_ANSWERS["general_help"])
    return {"answer": answer, "source": "policy_kb", "confidence": 1.0}


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
