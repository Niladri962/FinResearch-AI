"""Configurable reranking stage.

Providers
  fastembed              Cross-encoder on ONNX Runtime (default).
  sentence_transformers  Cross-encoder on PyTorch (optional dependency).
  lexical                Dependency-free term-coverage scorer.
  none                   Keep the hybrid-fusion order.

If a model-based reranker cannot be loaded (for example no network on first
run) the stage degrades to the lexical scorer instead of failing the query.
"""
from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod

from app.config import Settings
from app.utils.logging import get_logger
from app.utils.text import tokenize

logger = get_logger(__name__)


class Reranker(ABC):
    name: str = "abstract"

    @abstractmethod
    def score(self, query: str, documents: list[str]) -> list[float] | None:
        """Relevance score per document (higher is better); ``None`` keeps the incoming order."""


class NoopReranker(Reranker):
    name = "none"

    def score(self, query: str, documents: list[str]) -> list[float] | None:
        return None


class LexicalReranker(Reranker):
    """Query-term coverage with a bigram bonus and mild length normalisation."""

    name = "lexical"

    def score(self, query: str, documents: list[str]) -> list[float] | None:
        q_tokens = tokenize(query)
        if not q_tokens:
            return None
        q_set = set(q_tokens)
        q_bigrams = set(zip(q_tokens, q_tokens[1:]))
        scores: list[float] = []
        for document in documents:
            d_tokens = tokenize(document)
            if not d_tokens:
                scores.append(0.0)
                continue
            d_set = set(d_tokens)
            coverage = len(q_set & d_set) / len(q_set)
            bigram = len(q_bigrams & set(zip(d_tokens, d_tokens[1:]))) / len(q_bigrams) if q_bigrams else 0.0
            density = sum(1 for t in d_tokens if t in q_set) / len(d_tokens)
            scores.append(0.6 * coverage + 0.3 * bigram + 0.1 * min(1.0, density * 10))
        return scores


class _LazyModelReranker(Reranker):
    def __init__(self, model: str, cache_dir: str) -> None:
        self.name = model
        self._model_name = model
        self._cache_dir = cache_dir
        self._model = None
        self._failed = False
        self._lock = threading.Lock()
        self._fallback = LexicalReranker()

    def _load_model(self):  # noqa: ANN202
        raise NotImplementedError

    def _predict(self, model, query: str, documents: list[str]) -> list[float]:  # noqa: ANN001
        raise NotImplementedError

    def score(self, query: str, documents: list[str]) -> list[float] | None:
        if not documents:
            return []
        if not self._failed and self._model is None:
            with self._lock:
                if self._model is None and not self._failed:
                    try:
                        self._model = self._load_model()
                    except Exception as exc:
                        self._failed = True
                        self.name = f"lexical (fallback from {self._model_name})"
                        logger.warning(
                            "Reranker model unavailable, using lexical fallback",
                            extra={"model": self._model_name, "error": type(exc).__name__},
                        )
        if self._failed or self._model is None:
            return self._fallback.score(query, documents)
        try:
            raw = self._predict(self._model, query, documents)
            # Cross-encoders emit logits; squash to (0, 1) so thresholds are comparable.
            return [1.0 / (1.0 + math.exp(-max(min(float(s), 30.0), -30.0))) for s in raw]
        except Exception as exc:
            logger.warning("Reranking failed, using lexical fallback", extra={"error": type(exc).__name__})
            return self._fallback.score(query, documents)


class FastEmbedReranker(_LazyModelReranker):
    def _load_model(self):  # noqa: ANN202
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(model_name=self._model_name, cache_dir=self._cache_dir)

    def _predict(self, model, query: str, documents: list[str]) -> list[float]:  # noqa: ANN001
        return list(model.rerank(query, documents))


class SentenceTransformerReranker(_LazyModelReranker):
    def _load_model(self):  # noqa: ANN202
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self._model_name, cache_folder=self._cache_dir)

    def _predict(self, model, query: str, documents: list[str]) -> list[float]:  # noqa: ANN001
        return list(model.predict([(query, d) for d in documents]))


def build_reranker(settings: Settings) -> Reranker:
    provider = settings.reranker_provider
    cache_dir = str(settings.model_cache_dir)
    if provider == "none":
        return NoopReranker()
    if provider == "lexical":
        return LexicalReranker()
    if provider == "sentence_transformers":
        return SentenceTransformerReranker(settings.reranker_model, cache_dir)
    return FastEmbedReranker(settings.reranker_model, cache_dir)
