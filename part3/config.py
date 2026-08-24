"""Thresholds and mode switches for the support agent. Both threshold
constants below are measured, not guessed -- see
reports/part3_threshold_calibration.md and the intent-routing floor
discussion in part3/intent.py's module docstring.
"""
import os

GROUNDEDNESS_THRESHOLD = 0.50
INTENT_ROUTING_FLOOR = 0.25
RETRIEVAL_TOP_K = 3

# Below this, a message doesn't even resemble the conversational few-shot
# exemplars (greeting/help/thanks) -- so it's routed to a genuine "I can't
# help with that" answer instead of being silently absorbed into `policy`
# and producing a groundedness refusal that reads like a wall.
CONVERSATIONAL_ROUTING_FLOOR = 0.35

# A message is only treated as multi-clause/comparison if it splits into at
# least this many non-trivial clauses that each ground on a *different*
# document -- single-topic questions are unaffected.
MIN_CLAUSE_WORDS = 3

# A clause only counts toward "these clauses genuinely diverge onto
# different documents" if its own top match clears this floor -- otherwise
# a single coherent question that happens to contain "and" (e.g. "what
# happens if I open a pair of headphones and then decide I don't want
# them?") can get chopped into two weak, noisy half-questions that
# accidentally land on two different low-confidence documents and falsely
# trigger multi-clause mode. This is deliberately lower than
# GROUNDEDNESS_THRESHOLD -- it only gates *whether* to enter multi-clause
# mode, not whether a clause's final answer counts as grounded.
MULTI_CLAUSE_CONFIDENCE_FLOOR = 0.35

# A short pronoun-style follow-up ("What about if they're damaged?") is
# already retrieving the right document on its own text -- it's just
# phrased too tersely to clear the full groundedness bar. Empirically,
# rewriting the query by prepending the previous message tends to re-anchor
# retrieval onto the *previous* topic instead of the follow-up's own
# (correct) one, so instead this lower floor is used for a follow-up's own
# standalone match: still a real bar, just below GROUNDEDNESS_THRESHOLD,
# justified by conversation context (this is a continuation of an
# already-on-topic exchange, not a fresh out-of-domain message).
FOLLOWUP_GROUNDEDNESS_FLOOR = 0.35

PRODUCT_CATALOG_PATH = "data/product_catalog.json"
PRODUCT_INDEX_DIR = "data/product_index"

# MOCK_LLM is the default and the only mode every graded transcript in this
# repo is produced in: zero API keys, zero outbound network calls,
# deterministic. USE_LIVE_LLM is a strictly optional, never-scored extension.
USE_LIVE_LLM = os.environ.get("USE_LIVE_LLM", "") not in ("", "0", "false", "False")
LIVE_LLM_MODEL = os.environ.get("LIVE_LLM_MODEL", "llama3.2:3b")
LIVE_LLM_HOST = os.environ.get("LIVE_LLM_HOST", "http://localhost:11434")
