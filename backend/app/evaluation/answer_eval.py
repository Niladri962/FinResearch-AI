"""Generation metrics that need no judge model.

  faithfulness          share of answer statements supported by the retrieved context
  answer_relevance      how much of the question the answer addresses
  key_fact_recall       share of expected key facts present in the answer
  citation_correctness  share of cited sources that point at the relevant document/page

These are deterministic lexical/numeric proxies: cheap, reproducible and usable
in CI. ``ragas_eval`` adds LLM-judged versions when a judge model is available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import Citation
from app.utils.text import split_sentences, tokenize

_MARKUP = re.compile(r"\[[SCT]\d+(?:\s*,\s*[SCT]\d+)*\]|[*_#>`|]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _statements(answer: str) -> list[str]:
    statements: list[str] = []
    for line in answer.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "|", "_", ">")):
            continue  # headings, table rows, banners and notes are not claims
        clean = _MARKUP.sub(" ", line).strip(" -•")
        statements.extend(s for s in split_sentences(clean) if len(tokenize(s)) >= 3)
    return statements


def faithfulness(answer: str, contexts: list[str], *, threshold: float = 0.6) -> float:
    """A statement counts as supported when most of its content words and all of its
    figures occur in the combined context."""
    statements = _statements(answer)
    if not statements:
        return 0.0
    context_text = " ".join(contexts)
    context_tokens = set(tokenize(context_text))
    context_numbers = {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(context_text)}
    supported = 0
    for statement in statements:
        tokens = set(tokenize(statement))
        if not tokens:
            continue
        overlap = len(tokens & context_tokens) / len(tokens)
        numbers = {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(statement)}
        if overlap >= threshold and numbers <= context_numbers:
            supported += 1
    return supported / len(statements)


def answer_relevance(question: str, answer: str) -> float:
    terms = set(tokenize(question))
    if not terms:
        return 0.0
    return len(terms & set(tokenize(_MARKUP.sub(" ", answer)))) / len(terms)


def key_fact_recall(answer: str, key_facts: list[str]) -> float:
    if not key_facts:
        return 0.0
    haystack = re.sub(r"\s+", " ", answer.lower())
    return sum(1 for fact in key_facts if fact.lower() in haystack) / len(key_facts)


def citation_correctness(citations: list[Citation], relevant_document: str, relevant_page: int | None) -> tuple[float, float]:
    """Return ``(precision, page_hit)``.

    precision: share of citations that point at the expected document.
    page_hit:  1.0 when any citation covers the expected page.
    """
    if not citations:
        return 0.0, 0.0
    on_document = [c for c in citations if c.filename == relevant_document]
    page_hit = 0.0
    if relevant_page is not None:
        page_hit = 1.0 if any(c.page <= relevant_page <= (c.page_end or c.page) for c in on_document) else 0.0
    return len(on_document) / len(citations), page_hit


@dataclass
class AnswerScores:
    faithfulness: float
    answer_relevance: float
    key_fact_recall: float
    citation_precision: float
    citation_page_hit: float

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


def score_answer(
    *, question: str, answer: str, contexts: list[str], citations: list[Citation],
    key_facts: list[str], relevant_document: str, relevant_page: int | None,
) -> AnswerScores:
    precision, page_hit = citation_correctness(citations, relevant_document, relevant_page)
    return AnswerScores(
        faithfulness=faithfulness(answer, contexts),
        answer_relevance=answer_relevance(question, answer),
        key_fact_recall=key_fact_recall(answer, key_facts),
        citation_precision=precision,
        citation_page_hit=page_hit,
    )
