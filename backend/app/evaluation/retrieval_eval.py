"""Retrieval metrics: Recall@K, Precision@K, MRR and context relevance."""
from __future__ import annotations

from dataclasses import dataclass

from app.utils.text import tokenize


def recall_at_k(relevance: list[bool], total_relevant: int, k: int) -> float:
    """Share of all relevant items that appear in the top ``k`` results."""
    if total_relevant <= 0:
        return 0.0
    return min(sum(relevance[:k]), total_relevant) / total_relevant


def precision_at_k(relevance: list[bool], k: int) -> float:
    """Share of the top ``k`` results that are relevant (missing results count as misses)."""
    return sum(relevance[:k]) / k if k > 0 else 0.0


def reciprocal_rank(relevance: list[bool]) -> float:
    """1 / rank of the first relevant result, 0 if none is relevant."""
    for index, hit in enumerate(relevance, start=1):
        if hit:
            return 1.0 / index
    return 0.0


def hit_at_k(relevance: list[bool], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def context_relevance(query: str, contexts: list[str]) -> float:
    """Lexical proxy for context relevance: mean share of query terms present in each context."""
    terms = set(tokenize(query))
    if not terms or not contexts:
        return 0.0
    return sum(len(terms & set(tokenize(c))) / len(terms) for c in contexts) / len(contexts)


@dataclass
class RetrievalScores:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    hit_rate: float
    context_relevance: float

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


def score_retrieval(query: str, relevance: list[bool], contexts: list[str], *, total_relevant: int, k: int) -> RetrievalScores:
    return RetrievalScores(
        recall_at_k=recall_at_k(relevance, total_relevant, k),
        precision_at_k=precision_at_k(relevance, k),
        mrr=reciprocal_rank(relevance),
        hit_rate=hit_at_k(relevance, k),
        context_relevance=context_relevance(query, contexts),
    )


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
