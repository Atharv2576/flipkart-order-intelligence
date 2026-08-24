"""The LangGraph conversation state. Short-term memory (order_id,
order_features, turn_index, message history) is threaded explicitly through
each graph.invoke() call by the Conversation wrapper in agent.py -- there is
no module-level dict and no global, so a freshly constructed Conversation
genuinely starts with no memory of any other conversation.
"""
import operator
from typing import Annotated, Optional, TypedDict


class AgentState(TypedDict, total=False):
    user_message: str
    turn_index: int
    history: list[dict]

    order_id: Optional[int]
    order_features: Optional[dict]
    image_path: Optional[str]
    previous_message: Optional[str]
    previous_tool_error: Optional[str]

    blocked: bool
    blocked_patterns: list[str]

    intent: str
    intent_trace: dict

    retrieved_chunks: list[dict]
    retrieved_documents: list[dict]
    groundedness: dict

    tool_name: Optional[str]
    tool_result: Optional[dict]
    tool_error: Optional[str]
    missing_fields: Optional[list[str]]

    retrieval_mode: Optional[str]
    clause_evidence: Optional[list[dict]]
    structured_filters: Optional[dict]
    structured_filter_results: Optional[list[dict]]

    response: dict
    node_path: Annotated[list[str], operator.add]
