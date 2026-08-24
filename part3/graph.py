"""The agent's LangGraph: five nodes and one real conditional edge.

    START -> guard -> intent -> [conditional: route_by_intent]
                                  |-- blocked           -> respond
                                  |-- conversational     -> respond
                                  |-- policy             -> retrieve -> respond
                                  |-- return_risk         -> tool     -> respond
                                  |-- product_category    -> tool     -> respond
                                                                          respond -> END

The branch is genuinely load-bearing: a policy question never touches the
tool node, a return-risk question never runs retrieval, and a blocked
message reaches neither. `conversational` (greeting/help/thanks/farewell,
routed by the same nearest-few-shot-exemplar mechanism as the three
required intents -- see part3/intent.py) is the one addition: it answers
straight from response_node, the same way `blocked` already does, so a
plain "hi" no longer falls through to a policy-refusal message.
"""
from langgraph.graph import END, START, StateGraph

from part3.config import FOLLOWUP_GROUNDEDNESS_FLOOR, GROUNDEDNESS_THRESHOLD, RETRIEVAL_TOP_K, USE_LIVE_LLM
from part3.guardrails import scan_for_injection
from part3.intent import classify_intent, extract_order_id
from part3.mock_llm import (
    compose_blocked_answer,
    compose_comparison_answer,
    compose_conversational_answer,
    compose_missing_features_answer,
    compose_missing_input_answer,
    compose_policy_answer,
    compose_product_category_answer,
    compose_return_risk_answer,
)
from part3.query_decomposition import is_short_followup
from part3.retrieval import check_groundedness, retrieve_for_message, rollup_to_documents
from part3.slot_extraction import extract_order_features_from_text
from part3.state import AgentState
from part3.tools import check_return_risk, classify_product_image, lookup_order, missing_required_fields


def guard_node(state: AgentState) -> dict:
    result = scan_for_injection(state["user_message"])
    return {
        "blocked": result["blocked"],
        "blocked_patterns": result["matched_patterns"],
        "node_path": ["guard_node"],
    }


def _merge_extracted_features(order_features: dict | None, user_message: str) -> tuple[dict | None, bool]:
    """Fills in gaps in order_features from free text without ever
    overwriting an already-known value -- so "I paid using COD" in turn 2
    can't clobber a category already established in turn 1, and a full
    lookup_order() result is never partially replaced by a weaker guess.
    Returns (merged_features, gained_a_new_field) -- the second value is
    what lets intent_node recognise "It took 6 days to arrive" as still
    answering a pending return-risk clarification rather than a fresh,
    unrelated question.
    """
    extracted = extract_order_features_from_text(user_message)
    base = dict(order_features) if order_features else {}
    gained = False
    for key, value in extracted.items():
        if base.get(key) is None:
            base[key] = value
            gained = True
    return (base if base else order_features), gained


def intent_node(state: AgentState) -> dict:
    trace = classify_intent(state["user_message"])
    order_id = extract_order_id(state["user_message"]) or state.get("order_id")

    order_features = state.get("order_features")
    if order_id is not None and order_features is None:
        order_features = lookup_order(order_id)

    order_features, gained_new_field = _merge_extracted_features(order_features, state["user_message"])

    final_intent = trace["final_intent"]
    # A short reply that only supplies a missing order detail ("It took 6
    # days to arrive.") rarely resembles any return_risk few-shot exemplar
    # on its own, so classify_intent alone would route it as a fresh
    # (unrelated-looking) policy question. If the previous turn was
    # genuinely waiting on exactly this kind of answer and this message
    # contributed real data toward it, treat it as a continuation instead
    # of making the user repeat themselves or start over.
    if state.get("previous_tool_error") in ("missing_order_id", "missing_features") and (
        gained_new_field or order_id is not None
    ):
        final_intent = "return_risk"

    return {
        "intent": final_intent,
        "intent_trace": trace,
        "order_id": order_id,
        "order_features": order_features,
        "node_path": ["intent_node"],
    }


def route_by_intent(state: AgentState) -> str:
    if state["blocked"]:
        return "blocked"
    return state["intent"]


def retrieval_node(state: AgentState) -> dict:
    message = state["user_message"]

    from part3.product_catalog import is_structured_filter_query, parse_filter_query, query_products

    if is_structured_filter_query(message):
        filters = parse_filter_query(message)
        if filters:
            results = query_products(**filters)
            return {
                "retrieval_mode": "structured_filter",
                "structured_filters": filters,
                "structured_filter_results": results,
                "retrieved_documents": [],
                "groundedness": {"grounded": True, "best_score": 1.0, "threshold": GROUNDEDNESS_THRESHOLD},
                "node_path": ["retrieval_node"],
            }

    result = retrieve_for_message(message, top_k=RETRIEVAL_TOP_K)
    groundedness = check_groundedness(result["chunk_hits"], GROUNDEDNESS_THRESHOLD)

    is_followup = is_short_followup(message) and bool(state.get("previous_message"))
    if not groundedness["grounded"] and is_followup and groundedness["best_score"] >= FOLLOWUP_GROUNDEDNESS_FLOOR:
        groundedness = {**groundedness, "grounded": True}

    documents = rollup_to_documents(result["chunk_hits"])
    return {
        "retrieved_chunks": result["chunk_hits"],
        "retrieved_documents": documents,
        "groundedness": groundedness,
        "retrieval_mode": result["mode"],
        "clause_evidence": result["clause_evidence"],
        "node_path": ["retrieval_node"],
    }


def tool_node(state: AgentState) -> dict:
    if state["intent"] == "return_risk":
        order_features = state.get("order_features")
        if not order_features:
            return {"tool_name": "return_risk_tool", "tool_error": "missing_order_id", "node_path": ["tool_node"]}
        missing = missing_required_fields(order_features)
        if missing:
            return {
                "tool_name": "return_risk_tool",
                "tool_error": "missing_features",
                "missing_fields": missing,
                "node_path": ["tool_node"],
            }
        result = check_return_risk(order_features)
        return {"tool_name": "return_risk_tool", "tool_result": result, "node_path": ["tool_node"]}

    if state["intent"] == "product_category":
        image_path = state.get("image_path")
        if image_path is None:
            return {"tool_name": "image_classifier_tool", "tool_error": "missing_image", "node_path": ["tool_node"]}
        result = classify_product_image(image_path)
        return {"tool_name": "image_classifier_tool", "tool_result": result, "node_path": ["tool_node"]}

    return {"node_path": ["tool_node"]}


def response_node(state: AgentState) -> dict:
    if state["blocked"]:
        response = compose_blocked_answer(state["blocked_patterns"])
    elif state["intent"] == "conversational":
        response = compose_conversational_answer(state["intent_trace"].get("conversational_subtype"))
    elif state["intent"] == "policy":
        if state.get("retrieval_mode") == "structured_filter":
            from part3.product_catalog import compose_structured_filter_answer

            response = compose_structured_filter_answer(
                state["structured_filters"], state["structured_filter_results"]
            )
        elif state.get("retrieval_mode") == "multi":
            response = compose_comparison_answer(state["clause_evidence"], GROUNDEDNESS_THRESHOLD)
        else:
            response = compose_policy_answer(state.get("retrieved_documents", []), state["groundedness"])
    elif state["intent"] == "return_risk":
        if state.get("tool_error") == "missing_order_id":
            response = compose_missing_input_answer("order_id")
        elif state.get("tool_error") == "missing_features":
            response = compose_missing_features_answer(state["missing_fields"], state.get("order_features") or {})
        else:
            response = compose_return_risk_answer(
                state.get("order_id"), state["tool_result"], state.get("order_features")
            )
    elif state["intent"] == "product_category":
        if state.get("tool_error") == "missing_image":
            response = compose_missing_input_answer("image")
        else:
            response = compose_product_category_answer(state["tool_result"])
    else:
        response = compose_conversational_answer("general_help")

    if USE_LIVE_LLM and not state["blocked"]:
        from part3.live_llm import rewrite_with_live_llm

        evidence = ""
        if state.get("retrieved_documents"):
            evidence = "\n".join(d["chunk_text"] for d in state["retrieved_documents"])
        response = rewrite_with_live_llm(response, evidence)

    return {"response": response, "node_path": ["response_node"]}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("guard_node", guard_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("response_node", response_node)

    graph.add_edge(START, "guard_node")
    graph.add_edge("guard_node", "intent_node")
    graph.add_conditional_edges(
        "intent_node",
        route_by_intent,
        {
            "blocked": "response_node",
            "conversational": "response_node",
            "policy": "retrieval_node",
            "return_risk": "tool_node",
            "product_category": "tool_node",
        },
    )
    graph.add_edge("retrieval_node", "response_node")
    graph.add_edge("tool_node", "response_node")
    graph.add_edge("response_node", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_once(
    user_message: str,
    turn_index: int = 0,
    order_id: int | None = None,
    order_features: dict | None = None,
    image_path: str | None = None,
    previous_message: str | None = None,
    previous_tool_error: str | None = None,
) -> dict:
    """A single, stateless turn through the graph -- used by tests, the API,
    and the fresh-conversation transcripts. Multi-turn state is threaded by
    part3.agent.Conversation, not by this function.
    """
    graph = get_graph()
    initial_state: AgentState = {
        "user_message": user_message,
        "turn_index": turn_index,
        "order_id": order_id,
        "order_features": order_features,
        "image_path": image_path,
        "previous_message": previous_message,
        "previous_tool_error": previous_tool_error,
        "node_path": [],
    }
    final_state = graph.invoke(initial_state)
    final_state["trace"] = final_state["node_path"]
    final_state["doc_hits"] = final_state.get("retrieved_documents", [])
    return final_state
