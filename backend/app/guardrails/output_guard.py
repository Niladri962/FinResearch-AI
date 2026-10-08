"""Output guardrails: disclaimer, advice language and leak checks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DISCLAIMER = (
    "FinResearch AI is an analytical research assistant, not a financial advisor. Its output is generated from "
    "the uploaded documents and automated calculations, may contain errors, and should be independently verified. "
    "It is not investment advice and makes no guarantee about future performance."
)
INSUFFICIENT_EVIDENCE = (
    "I could not find sufficient evidence in the available documents to answer this reliably."
)

_ADVICE = [
    r"\b(you should|i recommend you|i advise you to|we recommend)\s+(definitely\s+)?(buy|sell|invest|hold|short)\b",
    r"\b(strong\s+)?(buy|sell)\s+(recommendation|rating)\b",
    r"\b(guaranteed|guarantee[sd]?|assured|risk[- ]free|sure[- ]shot|certain to)\b.{0,40}\b(returns?|profits?|gains?|rise|increase|upside)",
    r"\bwill\s+(definitely|certainly|surely|undoubtedly)\s+(rise|increase|grow|double|outperform|go up|fall|crash)",
    r"\b(can'?t|cannot)\s+lose\b|\bno[- ]risk\b",
]
_LEAK = [
    r"\b(gsk_|sk-|lsv2_)[A-Za-z0-9_\-]{16,}",
    r"(?i)\bmy (system )?(prompt|instructions) (is|are|says?)\b",
]
ADVICE_NOTE = (
    "\n\n> **Note:** Parts of this response may read as a recommendation or a statement of certainty. Treat them as "
    "interpretation of the documents, not as investment advice or a prediction."
)


@dataclass
class OutputGuardResult:
    text: str
    flags: list[str] = field(default_factory=list)


def check_output(text: str) -> OutputGuardResult:
    flags: list[str] = []
    for pattern in _LEAK:
        if re.search(pattern, text):
            flags.append("possible_leak")
            text = re.sub(pattern, "[redacted]", text)
    if any(re.search(p, text, re.I | re.S) for p in _ADVICE):
        flags.append("advice_language")
        text = text.rstrip() + ADVICE_NOTE
    return OutputGuardResult(text=text, flags=flags)
