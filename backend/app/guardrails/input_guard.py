"""Input guardrails.

Runs before any retrieval or LLM call. Outcomes:

  block    the request is refused with an explanation (prompt injection, market
           manipulation, clearly unrelated requests, malformed input)
  caution  the request proceeds, but the answer opens with a fixed statement
           (e.g. no guarantees about future returns)
  allow    normal processing
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier|preceding) (instructions?|prompts?|rules?|context)",
    r"disregard (all |any |the |your )?(previous|prior|above|earlier|system) (instructions?|prompts?|rules?)",
    r"forget (all |everything |your )?(previous |prior )?(instructions?|rules?|guidelines?)",
    r"(reveal|show|print|repeat|output|leak|display)\b.{0,40}\b(system|hidden|initial|developer) (prompt|instructions?|message)",
    r"what (is|are) your (system )?(prompt|instructions)",
    r"you are now\b|from now on you (are|will)|act as (an? )?(unrestricted|unfiltered|jailbroken)|\bdan mode\b|\bjailbreak\b",
    r"pretend (that )?(you|there) (are|have|is) no (rules|restrictions|guardrails|guidelines)",
    r"(override|bypass|disable|turn off) (your |the |all )?(safety|guardrails?|restrictions|filters?|rules)",
    r"<\s*/?\s*(system|assistant|instructions?)\s*>|\[\s*(system|inst)\s*\]",
    r"new instructions?:|system\s*:\s*you",
]
_MANIPULATION = [
    r"pump[- ]and[- ]dump|\bpump (the|this|a) (stock|share|price|coin)",
    r"(manipulate|rig|inflate|prop up)\b.{0,30}\b(stock|share|market|price)",
    r"insider (trading|information|tip)s?\b.{0,40}\b(how|use|profit|trade|get)|how to .{0,30}insider trad",
    r"(spread|create|write|start)\b.{0,30}\b(fake|false|misleading) (news|rumou?rs?|reports?|reviews?)",
    r"(falsify|fabricate|cook|doctor|fake)\b.{0,30}\b(accounts?|books|financial statements?|numbers|earnings|report)",
    r"(evade|avoid getting caught|hide .{0,20} from)\b.{0,30}\b(regulator|sec|sebi|auditor|tax)",
    r"front[- ]?run|wash trad|spoof(ing)? (orders?|the market)",
]
_GUARANTEE = [
    r"\b(guarantee[ds]?|guaranteed|assured|sure[- ]?shot|risk[- ]free|can'?t lose|cannot lose|100\s*%|certain(ly)?)\b.{0,60}\b(returns?|profits?|gains?|upside|multibagger|double|increase|rise|go up)",
    r"\b(will|going to)\s+(definitely|surely|certainly|absolutely)\b",
    r"\bdefinitely\s+(increase|rise|go up|double|grow|fall|crash|outperform|beat)",
    r"which (stock|share|company|fund)s?\b.{0,60}\b(will|is going to|to)\b.{0,30}\b(increase|rise|go up|double|triple|multiply|\d+\s*(%|x|percent))",
    r"\b(should i|shall i|do i)\s+(buy|sell|invest|hold|short)\b",
    r"\b(tell me|give me)\b.{0,30}\b(what|which)\b.{0,20}\b(to buy|to sell|to invest)",
    r"\b(price target|target price)\b.{0,30}\b(next|tomorrow|week|month)",
    r"\bmultibagger\b|\bget rich\b|\bdouble my money\b",
]
_OFF_TOPIC = [
    r"\b(write|compose|generate)\b.{0,20}\b(poem|song|lyrics|haiku|limerick|story|joke|essay on)\b",
    r"\b(recipe|cook|bake)\b|\bweather (in|for|today|tomorrow)\b|\bhoroscope\b",
    r"\b(movie|film|tv show|series|game|anime) (recommendation|review|to watch)",
    r"\b(write|debug|fix)\b.{0,30}\b(python|javascript|java|c\+\+|sql|html) (code|script|function|program)\b",
    r"\b(translate|translation)\b.{0,30}\b(into|to) (french|spanish|german|hindi|chinese|japanese)\b",
    r"\b(medical|legal) advice\b|\bdiagnos(e|is)\b|\bsymptoms? of\b",
    r"\bwho (won|will win) the\b|\bcapital of\b|\btell me a joke\b",
]
_FINANCE_HINT = re.compile(
    r"(?i)\b(revenue|profit|margin|debt|equity|cash|ebitda|ebit|eps|earnings|ratio|balance sheet|income|"
    r"financial|fiscal|quarter|annual|company|shareholder|dividend|capex|assets?|liabilit|risk|management|"
    r"growth|valuation|stock|share|market|invest|report|filing|fy\d{2,4}|guidance|outlook|sales|cost)\w*"
)

GUARANTEE_NOTICE = (
    "**I can't predict or guarantee future returns.** No analysis can tell you with certainty how a stock will "
    "perform, and I don't make buy, sell or hold recommendations. What I can do is summarise what the uploaded "
    "documents show about the company's performance, outlook and risks so you can form your own view."
)
INJECTION_REFUSAL = (
    "I can't follow instructions that try to change how I operate or reveal my configuration. "
    "I'm happy to help with questions about the financial documents you've uploaded."
)
MANIPULATION_REFUSAL = (
    "I can't help with market manipulation, insider trading, misleading disclosures or evading regulators. "
    "I can help you analyse a company's reported financials, risks and management commentary."
)
OFF_TOPIC_REFUSAL = (
    "I'm a financial research assistant, so I can only help with questions about companies, financial "
    "documents and financial analysis. Try asking about revenue trends, ratios, risks or management commentary."
)


@dataclass
class GuardResult:
    action: str = "allow"                 # allow | caution | block
    categories: list[str] = field(default_factory=list)
    message: str = ""                     # refusal text (block) or opening notice (caution)

    @property
    def blocked(self) -> bool:
        return self.action == "block"


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.I | re.S) for p in patterns)


def check_input(query: str, *, max_chars: int = 2000) -> GuardResult:
    text = query.strip()
    if not text:
        return GuardResult("block", ["empty"], "Please enter a question.")
    if len(text) > max_chars:
        return GuardResult("block", ["too_long"], f"Your question is too long. Please keep it under {max_chars} characters.")
    if sum(1 for ch in text if ch.isprintable() or ch in "\n\t") / len(text) < 0.9:
        return GuardResult("block", ["malformed"], "Your question contains unsupported characters.")

    if _any(_INJECTION, text):
        return GuardResult("block", ["prompt_injection"], INJECTION_REFUSAL)
    if _any(_MANIPULATION, text):
        return GuardResult("block", ["market_manipulation"], MANIPULATION_REFUSAL)
    if _any(_OFF_TOPIC, text) and not _FINANCE_HINT.search(text):
        return GuardResult("block", ["off_topic"], OFF_TOPIC_REFUSAL)
    if _any(_GUARANTEE, text):
        return GuardResult("caution", ["guaranteed_returns"], GUARANTEE_NOTICE)
    return GuardResult()


def is_suspicious_context(text: str) -> bool:
    """Detect instructions planted inside a document (indirect prompt injection)."""
    return _any(_INJECTION, text)
