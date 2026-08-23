"""The agent's LangGraph: five nodes and one real conditional edge.

    START -> guard -> intent -> [conditional: route_by_intent]
                                  |-- blocked           -> respond
                                  |-- policy             -> retrieve -> respond
                                  |-- return_risk         -> tool     -> respond
                                  |-- product_category    -> tool     -> respond
                                                                          respond -> END

The branch is genuinely load-bearing: a policy question never touches the
tool node, a return-risk question never runs retrieval, and a blocked
message reaches neither.
"""
from langgraph.graph import END, START, StateGraph

from part3.config import GROUNDEDNESS_THRESHOLD, RETRIEVAL_TOP_K, USE_LIVE_LLM
from part3.guardrails import scan_for_injection
from part3.intent import classify_intent, extract_order_id
from part3.mock_llm import (
    compose_blocked_answer,
    compose_missing_input_answer,
    compose_policy_answer,
    compose_product_category_answer,
    compose_return_risk_answer,
)
from part3.retrieval import check_groundedness, rollup_to_documents, search_chunks
from part3.state import AgentState
from part3.tools import check_return_risk, classify_product_image, lookup_order


def guard_node(state: AgentState) -> dict:
    result = scan_for_injection(state["user_message"])
    return {
        "blocked": result["blocked"],
        "blocked_patterns": result["matched_patterns"],
        "node_path": ["guard_node"],
    }


def intent_node(state: AgentState) -> dict:
    trace = classify_intent(state["user_message"])
    order_id = extract_order_id(state["user_message"]) or state.get("order_id")

    order_features = state.get("order_features")
    if order_id is not None and order_features is None:
        order_features = lookup_order(order_id)

    return {
        "intent": trace["final_intent"],
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
    chunk_hits = search_chunks(state["user_message"], top_k=RETRIEVAL_TOP_K)
    documents = rollup_to_documents(chunk_hits)
    groundedness = check_groundedness(chunk_hits, GROUNDEDNESS_THRESHOLD)
    return {
        "retrieved_chunks": chunk_hits,
        "retrieved_documents": documents,
        "groundedness": groundedness,
        "node_path": ["retrieval_node"],
    }


def tool_node(state: AgentState) -> dict:
    if state["intent"] == "return_risk":
        order_features = state.get("order_features")
        if order_features is None:
            return {"tool_name": "return_risk_tool", "tool_error": "missing_order_id", "node_path": ["tool_node"]}
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
    elif state["intent"] == "policy":
        response = compose_policy_answer(state.get("retrieved_documents", []), state["groundedness"])
    elif state["intent"] == "return_risk":
        if state.get("tool_error") == "missing_order_id":
            response = compose_missing_input_answer("order_id")
        else:
            response = compose_return_risk_answer(state.get("order_id"), state["tool_result"])
    elif state["intent"] == "product_category":
        if state.get("tool_error") == "missing_image":
            response = compose_missing_input_answer("image")
        else:
            response = compose_product_category_answer(state["tool_result"])
    else:
        response = compose_blocked_answer([])

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
        "node_path": [],
    }
    final_state = graph.invoke(initial_state)
    final_state["trace"] = final_state["node_path"]
    final_state["doc_hits"] = final_state.get("retrieved_documents", [])
    return final_state
