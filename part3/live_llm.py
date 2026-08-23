"""OPTIONAL live-LLM extension, active only when USE_LIVE_LLM=1. Never
required, never used by any graded transcript, and never a substitute for
MOCK_LLM mode -- removing the flag (the default) leaves every acceptance
criterion satisfied through MOCK_LLM alone.

Calls a small local model through Ollama (http://localhost:11434), which
needs no API key and no outbound internet access once the model is pulled.
It can only *rephrase* the deterministic MOCK_LLM answer using the same
retrieved evidence; it never introduces a fact, a source, or a confidence
value MOCK_LLM did not already compute, and any failure -- Ollama not
running, the model missing, a timeout -- falls back to the unmodified
MOCK_LLM answer.
"""
import json

import requests

from part3.config import LIVE_LLM_HOST, LIVE_LLM_MODEL
from part3.prompts import SYSTEM_PROMPT

REWRITE_TIMEOUT_SECONDS = 15


def rewrite_with_live_llm(mock_response: dict, evidence_text: str) -> dict:
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Evidence you may use:\n{evidence_text}\n\n"
        f"Deterministic answer to rephrase (do not add or remove any fact, "
        f"number, or claim -- only improve the phrasing):\n{mock_response['answer']}\n\n"
        "Rewritten answer (plain text, at most three sentences, no JSON):"
    )

    try:
        response = requests.post(
            f"{LIVE_LLM_HOST}/api/generate",
            json={"model": LIVE_LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=REWRITE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        rewritten_text = json.loads(response.text)["response"].strip()
        if not rewritten_text:
            return mock_response
        return {**mock_response, "answer": rewritten_text, "rephrased_by": LIVE_LLM_MODEL}
    except Exception as exc:  # noqa: BLE001 -- any failure at all falls back to the mock answer
        return {**mock_response, "live_llm_error": str(exc)}
