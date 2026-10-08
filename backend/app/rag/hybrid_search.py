"""Fusion of semantic and keyword result lists.

Two strategies:

  weighted  min-max normalise each list, then ``w_sem * semantic + w_kw * keyword``.
  rrf       reciprocal-rank fusion — rank-based, robust when score scales are unrelated.
"""
from __future__ import annotations

from dataclasses import dataclass

RRF_K = 60


@dataclass
class FusedCandidate:
    chunk_id: str
    semantic_score: float = 0.0   # raw cosine similarity
    keyword_score: float = 0.0    # raw BM25
    hybrid_score: float = 0.0


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high - low < 1e-12:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def fuse(
    semantic: list[tuple[str, float]],
    keyword: list[tuple[str, float]],
    *,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    method: str = "weighted",
) -> list[FusedCandidate]:
    """Merge two ranked lists of ``(chunk_id, score)`` into one, best first."""
    sem, kw = dict(semantic), dict(keyword)
    candidates = {cid: FusedCandidate(cid, sem.get(cid, 0.0), kw.get(cid, 0.0)) for cid in {*sem, *kw}}

    if method == "rrf":
        for weight, ranked in ((semantic_weight, semantic), (keyword_weight, keyword)):
            for rank, (cid, _score) in enumerate(ranked, start=1):
                candidates[cid].hybrid_score += weight / (RRF_K + rank)
    else:
        total = (semantic_weight + keyword_weight) or 1.0
        norm_sem, norm_kw = _min_max(sem), _min_max(kw)
        for cid, candidate in candidates.items():
            candidate.hybrid_score = (
                semantic_weight * norm_sem.get(cid, 0.0) + keyword_weight * norm_kw.get(cid, 0.0)
            ) / total

    return sorted(candidates.values(), key=lambda c: (-c.hybrid_score, c.chunk_id))
