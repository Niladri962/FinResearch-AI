"""Configurable embedding layer.

Providers
  fastembed              BGE (and other) models on ONNX Runtime — default, no PyTorch.
  sentence_transformers  Any Sentence-Transformers model (optional dependency).
  openai                 Any OpenAI-compatible ``/embeddings`` endpoint.
  hash                   Deterministic feature hashing — offline, for tests and air-gapped demos.

All providers return L2-normalised vectors so cosine similarity is a dot product.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import threading
from abc import ABC, abstractmethod

import httpx

from app.config import Settings
from app.utils.errors import EmbeddingError
from app.utils.logging import get_logger
from app.utils.text import tokenize

logger = get_logger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class Embedder(ABC):
    name: str = "abstract"

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder(Embedder):
    """Signed feature hashing over unigrams and bigrams.

    Not a semantic model — it captures lexical overlap only — but it is fast,
    dependency-free and fully deterministic, which makes pipeline tests reliable.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension
        self.name = f"hash-{dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vector = [0.0] * self._dimension
        for feature in features:
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        # Sub-linear term frequency keeps long chunks from dominating.
        vector = [math.copysign(math.sqrt(abs(v)), v) for v in vector]
        return _normalize(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbedder(Embedder):
    def __init__(self, model: str, cache_dir: str, batch_size: int = 32) -> None:
        self.name = model
        self._model_name = model
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._model = None
        self._dimension: int | None = None
        self._lock = threading.Lock()

    def _load(self):  # noqa: ANN202
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from fastembed import TextEmbedding

                        self._model = TextEmbedding(model_name=self._model_name, cache_dir=self._cache_dir)
                    except Exception as exc:
                        raise EmbeddingError(
                            f"Could not load embedding model '{self._model_name}'. Check network access for the "
                            "first download or set EMBEDDING_PROVIDER=hash to run offline."
                        ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._load().passage_embed(texts, batch_size=self._batch_size)
            return [_normalize([float(x) for x in v]) for v in vectors]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Embedding generation failed.") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            vector = next(iter(self._load().query_embed(text)))
            return _normalize([float(x) for x in vector])
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Query embedding failed.") from exc


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str, cache_dir: str, batch_size: int = 32) -> None:
        self.name = model
        self._model_name = model
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()
        # BGE v1.5 English models expect an instruction on the query side only.
        self._query_prefix = _BGE_QUERY_PREFIX if "bge" in model.lower() and "m3" not in model.lower() else ""

    def _load(self):  # noqa: ANN202
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise EmbeddingError(
                            "sentence-transformers is not installed. Run "
                            "`pip install -r requirements-optional.txt` or use EMBEDDING_PROVIDER=fastembed."
                        ) from exc
                    try:
                        self._model = SentenceTransformer(self._model_name, cache_folder=self._cache_dir)
                    except Exception as exc:
                        raise EmbeddingError(f"Could not load embedding model '{self._model_name}'.") from exc
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._load().encode(texts, batch_size=self._batch_size, normalize_embeddings=True)
            return [[float(x) for x in v] for v in vectors]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Embedding generation failed.") from exc

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self._query_prefix + text])[0]


class OpenAICompatibleEmbedder(Embedder):
    def __init__(self, model: str, api_key: str, base_url: str, batch_size: int = 64, timeout: float = 60.0) -> None:
        if not api_key:
            raise EmbeddingError("EMBEDDING_API_KEY is required for EMBEDDING_PROVIDER=openai.")
        self.name = model
        self._model = model
        self._batch_size = batch_size
        self._client = httpx.Client(
            base_url=(base_url or "https://api.openai.com/v1").rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start:start + self._batch_size]
            try:
                response = self._client.post("/embeddings", json={"model": self._model, "input": batch})
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: item["index"])
            except httpx.HTTPStatusError as exc:
                raise EmbeddingError(f"Embedding API returned HTTP {exc.response.status_code}.") from exc
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                raise EmbeddingError("Embedding API request failed.") from exc
            vectors.extend(_normalize([float(x) for x in item["embedding"]]) for item in data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def local_models_available() -> bool:
    """False on slim installs (e.g. serverless) that omit the ONNX model runtime."""
    return importlib.util.find_spec("fastembed") is not None


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider
    cache_dir = str(settings.model_cache_dir)
    if provider == "fastembed" and not local_models_available():
        # Slim deployment without local models: use an embeddings API when one is
        # configured, otherwise the built-in lexical embedder so the app still works.
        if settings.embedding_api_key:
            logger.warning("fastembed is not installed; using the configured embeddings API")
            model = settings.embedding_model if "/" not in settings.embedding_model else "text-embedding-3-small"
            return OpenAICompatibleEmbedder(model, settings.embedding_api_key, settings.embedding_base_url)
        logger.warning("fastembed is not installed and no EMBEDDING_API_KEY is set; using the hash embedder")
        return HashEmbedder(settings.hash_embedding_dim)
    if provider == "hash":
        return HashEmbedder(settings.hash_embedding_dim)
    if provider == "fastembed":
        return FastEmbedEmbedder(settings.embedding_model, cache_dir, settings.embedding_batch_size)
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.embedding_model, cache_dir, settings.embedding_batch_size)
    if provider == "openai":
        return OpenAICompatibleEmbedder(
            settings.embedding_model,
            settings.embedding_api_key or settings.llm_api_key,
            settings.embedding_base_url,
        )
    raise EmbeddingError(f"Unknown embedding provider '{provider}'.")
