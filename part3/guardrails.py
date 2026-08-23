"""Input-side prompt-injection filtering. This scan runs on the raw user
text before any node does real work, deliberately outside the model: a
model that has already been successfully injected is exactly the component
you can no longer trust to report the injection itself. Patterns require
the surrounding override structure (an instruction word plus a target like
"instructions", "prompt", or "rules"), not a single word in isolation, so an
ordinary question like "Can I ignore the delivery SMS?" is not blocked.
"""
import re

INJECTION_PATTERNS = [
    r"ignore (all|any|the|previous|prior|above)\b.{0,20}\b(instructions?|rules?|prompt)",
    r"disregard\b.{0,20}\b(instructions?|rules?|prompt|guidelines?)",
    r"forget (your |all |the )?(previous |prior )?instructions?",
    r"(pretend (that )?you (are|were)|act as if you (are|were))\b.{0,40}\b"
    r"(not (an? )?ai|no (rules|restrictions|filters)|without (rules|restrictions|filters|limits)|"
    r"unrestricted|unfiltered|\bdan\b|developer mode)",
    r"you are now (in )?(developer|dan|jailbreak) mode",
    r"reveal (your |the )?(system )?prompt",
    r"(print|show|output) (your |the )?(system )?(prompt|instructions)",
    r"bypass (your |the )?(safety|guidelines?|restrictions?|rules?)",
    r"override (your |the )?(instructions?|rules?|configuration)",
    r"do anything now",
    r"jailbreak",
    r"\bdan\s*mode\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def scan_for_injection(text: str) -> dict:
    matched = [p for p, compiled in zip(INJECTION_PATTERNS, _COMPILED) if compiled.search(text)]
    return {
        "blocked": len(matched) > 0,
        "patterns_checked": len(INJECTION_PATTERNS),
        "matched_patterns": matched,
    }
