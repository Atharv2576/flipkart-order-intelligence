"""Thresholds and mode switches for the support agent. Both threshold
constants below are measured, not guessed -- see
reports/part3_threshold_calibration.md and the intent-routing floor
discussion in part3/intent.py's module docstring.
"""
import os

GROUNDEDNESS_THRESHOLD = 0.50
INTENT_ROUTING_FLOOR = 0.25
RETRIEVAL_TOP_K = 3

# MOCK_LLM is the default and the only mode every graded transcript in this
# repo is produced in: zero API keys, zero outbound network calls,
# deterministic. USE_LIVE_LLM is a strictly optional, never-scored extension.
USE_LIVE_LLM = os.environ.get("USE_LIVE_LLM", "") not in ("", "0", "false", "False")
LIVE_LLM_MODEL = os.environ.get("LIVE_LLM_MODEL", "llama3.2:3b")
LIVE_LLM_HOST = os.environ.get("LIVE_LLM_HOST", "http://localhost:11434")
