"""Text helpers shared by chunking, BM25, guardrails and evaluation."""
from __future__ import annotations

import hashlib
import re

STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its may me my of on or our out over shall should so than that the their them then
    there these they this those to under up us was we were what when where which who whom why will
    with would you your about above after again all also am any because before being below between
    both down during each few further here more most no nor not off once only other own same some
    such too very s t just now per please tell show give find""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_FY_RE = re.compile(r"^fy(\d{2}|\d{4})$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WORD_RE = re.compile(r"\w+|[^\w\s]")
_WS_RE = re.compile(r"[ \t ]+")


def _stem(token: str) -> str:
    """Very light plural stemmer — enough to match 'risks'/'risk', 'margins'/'margin'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase lexical tokens tuned for financial text.

    Hyphenated terms are split ("debt-to-equity" → debt, equity) and fiscal-year
    tokens are expanded ("FY25" → fy2025, 2025) so period mentions match regardless
    of how a filing formats them.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        fy = _FY_RE.match(raw)
        if fy:
            year = fy.group(1)
            year = year if len(year) == 4 else f"20{year}"
            tokens.extend((f"fy{year}", year))
            continue
        if drop_stopwords and raw in STOPWORDS:
            continue
        tokens.append(_stem(raw))
    return tokens


def count_tokens(text: str) -> int:
    """Cheap model-agnostic token estimate (words + punctuation)."""
    return len(_WORD_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def normalize_whitespace(text: str) -> str:
    text = _WS_RE.sub(" ", text.replace("\r", ""))
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "item"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
