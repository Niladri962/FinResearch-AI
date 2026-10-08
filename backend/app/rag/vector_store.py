"""Vector store abstraction.

``VectorStore`` is the only interface the rest of the app sees, so the database
is replaceable. Implementations:

  QdrantVectorStore   Qdrant server / Qdrant Cloud through its REST API.
  LocalVectorStore    Exact cosine search with NumPy, persisted to disk. Zero-infra default.
  (memory)            ``LocalVectorStore`` without a path — used in tests.
"""
from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from sqlalchemy import delete, func, select

from app.config import Settings
from app.models.database import SessionFactory, session_scope
from app.models.db import ChunkVector
from app.utils.errors import VectorStoreError
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorPoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class VectorHit:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchFilter:
    """Metadata filter. Empty lists mean "no restriction"; fields are AND-ed, values OR-ed."""

    company_ids: list[int] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    fiscal_years: list[int] = field(default_factory=list)
    document_types: list[str] = field(default_factory=list)

    def clauses(self) -> dict[str, list[Any]]:
        return {
            key: values
            for key, values in (
                ("company_id", self.company_ids),
                ("document_id", self.document_ids),
                ("fiscal_year", self.fiscal_years),
                ("document_type", self.document_types),
            )
            if values
        }

    def matches(self, payload: dict[str, Any]) -> bool:
        return all(payload.get(key) in values for key, values in self.clauses().items())


class VectorStore(ABC):
    backend: str = "abstract"

    @abstractmethod
    def ensure_collection(self, dimension: int) -> None: ...

    @abstractmethod
    def upsert(self, points: list[VectorPoint]) -> None: ...

    @abstractmethod
    def search(self, vector: list[float], limit: int, flt: SearchFilter | None = None) -> list[VectorHit]: ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    def healthy(self) -> bool:
        try:
            self.count()
            return True
        except Exception:
            return False


class LocalVectorStore(VectorStore):
    """Brute-force cosine search. Exact, and fast enough for tens of thousands of chunks."""

    def __init__(self, path: Path | None = None) -> None:
        self.backend = "local" if path else "memory"
        self._path = path
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._payloads: list[dict[str, Any]] = []
        self._matrix: np.ndarray | None = None
        self._dimension: int | None = None
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            self._load()

    # ── Persistence ──────────────────────────────────────────────────────
    def _load(self) -> None:
        assert self._path is not None
        meta, vectors = self._path / "index.json", self._path / "vectors.npy"
        if not (meta.exists() and vectors.exists()):
            return
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            matrix = np.load(vectors)
            if len(data["ids"]) != matrix.shape[0]:
                raise ValueError("index and vector files are out of sync")
            self._ids, self._payloads = data["ids"], data["payloads"]
            self._dimension = data.get("dimension")
            self._matrix = matrix if matrix.size else None
        except Exception as exc:
            raise VectorStoreError(
                "The local vector index is corrupted. Delete the DATA_DIR/vectors folder and re-upload documents."
            ) from exc

    def _save(self) -> None:
        if self._path is None:
            return
        matrix = self._matrix if self._matrix is not None else np.zeros((0, self._dimension or 0), dtype=np.float32)
        tmp_vectors, tmp_meta = self._path / "vectors.tmp.npy", self._path / "index.tmp.json"
        np.save(tmp_vectors, matrix)
        tmp_meta.write_text(
            json.dumps({"dimension": self._dimension, "ids": self._ids, "payloads": self._payloads}),
            encoding="utf-8",
        )
        tmp_vectors.replace(self._path / "vectors.npy")
        tmp_meta.replace(self._path / "index.json")

    # ── Interface ────────────────────────────────────────────────────────
    def ensure_collection(self, dimension: int) -> None:
        with self._lock:
            if self._dimension is None or not self._ids:
                self._dimension = dimension
            elif self._dimension != dimension:
                raise VectorStoreError(
                    f"The index was built with {self._dimension}-dimensional embeddings but the configured model "
                    f"produces {dimension}. Re-index your documents after changing EMBEDDING_MODEL."
                )

    def upsert(self, points: list[VectorPoint]) -> None:
        if not points:
            return
        with self._lock:
            self.ensure_collection(len(points[0].vector))
            incoming = {p.id for p in points}
            if incoming & set(self._ids):
                self._remove([i for i, pid in enumerate(self._ids) if pid in incoming])
            new = np.asarray([p.vector for p in points], dtype=np.float32)
            self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])
            self._ids.extend(p.id for p in points)
            self._payloads.extend(p.payload for p in points)
            self._save()

    def _remove(self, indices: list[int]) -> None:
        if not indices or self._matrix is None:
            return
        keep = np.ones(len(self._ids), dtype=bool)
        keep[indices] = False
        self._matrix = self._matrix[keep]
        self._ids = [pid for pid, k in zip(self._ids, keep) if k]
        self._payloads = [pl for pl, k in zip(self._payloads, keep) if k]
        if not self._ids:
            self._matrix = None

    def search(self, vector: list[float], limit: int, flt: SearchFilter | None = None) -> list[VectorHit]:
        with self._lock:
            if self._matrix is None or not self._ids:
                return []
            query = np.asarray(vector, dtype=np.float32)
            if query.shape[0] != self._matrix.shape[1]:
                raise VectorStoreError("Query embedding dimension does not match the index.")
            scores = self._matrix @ query
            if flt and flt.clauses():
                mask = np.fromiter((flt.matches(p) for p in self._payloads), dtype=bool, count=len(self._payloads))
                scores = np.where(mask, scores, -np.inf)
            order = np.argsort(-scores)[:limit]
            return [
                VectorHit(id=self._ids[i], score=float(scores[i]), payload=self._payloads[i])
                for i in order
                if np.isfinite(scores[i])
            ]

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self._remove([i for i, p in enumerate(self._payloads) if p.get("document_id") == document_id])
            self._save()

    def count(self) -> int:
        return len(self._ids)


class QdrantVectorStore(VectorStore):
    """Qdrant over REST. Works with a self-hosted server and with Qdrant Cloud."""

    backend = "qdrant"
    _INDEXED_FIELDS = {"company_id": "integer", "document_id": "keyword", "fiscal_year": "integer", "document_type": "keyword"}

    def __init__(self, url: str, api_key: str = "", collection: str = "finresearch_chunks", timeout: float = 30.0) -> None:
        headers = {"api-key": api_key} if api_key else {}
        self._client = httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=timeout)
        self._collection = collection
        self._ready = False

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise VectorStoreError("The vector database is unreachable. Check QDRANT_URL and that Qdrant is running.") from exc
        if response.status_code >= 400:
            if response.status_code in (401, 403):
                raise VectorStoreError("The vector database rejected the credentials. Check QDRANT_API_KEY.")
            raise VectorStoreError(f"The vector database returned HTTP {response.status_code}.")
        return response.json()

    def ensure_collection(self, dimension: int) -> None:
        if self._ready:
            return
        try:
            response = self._client.get(f"/collections/{self._collection}")
        except httpx.HTTPError as exc:
            raise VectorStoreError("The vector database is unreachable. Check QDRANT_URL and that Qdrant is running.") from exc
        if response.status_code == 404:
            self._request(
                "PUT", f"/collections/{self._collection}",
                json={"vectors": {"size": dimension, "distance": "Cosine"}},
            )
            for name, schema in self._INDEXED_FIELDS.items():
                self._request(
                    "PUT", f"/collections/{self._collection}/index", params={"wait": "true"},
                    json={"field_name": name, "field_schema": schema},
                )
        elif response.status_code >= 400:
            raise VectorStoreError(f"The vector database returned HTTP {response.status_code}.")
        else:
            vectors = response.json().get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
            existing = vectors.get("size")
            if existing and existing != dimension:
                raise VectorStoreError(
                    f"Collection '{self._collection}' holds {existing}-dimensional vectors but the configured "
                    f"embedding model produces {dimension}. Use a new QDRANT_COLLECTION or re-index."
                )
        self._ready = True

    @staticmethod
    def _filter(flt: SearchFilter | None) -> dict[str, Any] | None:
        if not flt or not flt.clauses():
            return None
        return {"must": [{"key": key, "match": {"any": values}} for key, values in flt.clauses().items()]}

    def upsert(self, points: list[VectorPoint]) -> None:
        if not points:
            return
        self.ensure_collection(len(points[0].vector))
        for start in range(0, len(points), 128):
            batch = points[start:start + 128]
            self._request(
                "PUT", f"/collections/{self._collection}/points", params={"wait": "true"},
                json={"points": [{"id": p.id, "vector": p.vector, "payload": p.payload} for p in batch]},
            )

    def search(self, vector: list[float], limit: int, flt: SearchFilter | None = None) -> list[VectorHit]:
        self.ensure_collection(len(vector))
        body: dict[str, Any] = {"query": vector, "limit": limit, "with_payload": True}
        if (query_filter := self._filter(flt)) is not None:
            body["filter"] = query_filter
        result = self._request("POST", f"/collections/{self._collection}/points/query", json=body)
        points = result.get("result", {}).get("points", [])
        return [VectorHit(id=str(p["id"]), score=float(p["score"]), payload=p.get("payload") or {}) for p in points]

    def delete_document(self, document_id: str) -> None:
        if not self._collection_exists():
            return
        self._request(
            "POST", f"/collections/{self._collection}/points/delete", params={"wait": "true"},
            json={"filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]}},
        )

    def _collection_exists(self) -> bool:
        try:
            return self._client.get(f"/collections/{self._collection}").status_code == 200
        except httpx.HTTPError as exc:
            raise VectorStoreError("The vector database is unreachable.") from exc

    def count(self) -> int:
        if not self._collection_exists():
            return 0
        result = self._request("POST", f"/collections/{self._collection}/points/count", json={"exact": True})
        return int(result.get("result", {}).get("count", 0))


class SqlVectorStore(VectorStore):
    """Vectors stored in the relational database, searched exactly in memory.

    For deployments with no persistent disk and no vector database (serverless).
    The in-memory matrix is rebuilt whenever the stored set changes, detected by a
    cheap count/latest-timestamp signature, so separate function instances stay
    consistent without coordinating.
    """

    backend = "database"

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory
        self._lock = threading.Lock()
        self._signature: tuple | None = None
        self._cache = LocalVectorStore(None)

    def _current_signature(self, session) -> tuple:  # noqa: ANN001
        count, latest = session.execute(select(func.count(ChunkVector.chunk_id), func.max(ChunkVector.created_at))).one()
        return (count, str(latest))

    def _refresh(self) -> LocalVectorStore:
        try:
            with self._session_factory() as session:
                signature = self._current_signature(session)
                if signature == self._signature:
                    return self._cache
                with self._lock:
                    rows = session.execute(select(ChunkVector.chunk_id, ChunkVector.vector, ChunkVector.payload)).all()
                    cache = LocalVectorStore(None)
                    cache.upsert([
                        VectorPoint(chunk_id, np.frombuffer(blob, dtype="<f4").tolist(), payload or {})
                        for chunk_id, blob, payload in rows
                    ])
                    self._cache, self._signature = cache, signature
                return self._cache
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("The vector index could not be read from the database.") from exc

    def ensure_collection(self, dimension: int) -> None:
        with self._session_factory() as session:
            existing = session.execute(select(ChunkVector.dimension).limit(1)).scalar_one_or_none()
        if existing is not None and existing != dimension:
            raise VectorStoreError(
                f"The index was built with {existing}-dimensional embeddings but the configured model "
                f"produces {dimension}. Delete the documents and upload them again after changing EMBEDDING_MODEL."
            )

    def upsert(self, points: list[VectorPoint]) -> None:
        if not points:
            return
        self.ensure_collection(len(points[0].vector))
        try:
            with session_scope(self._session_factory) as session:
                session.execute(delete(ChunkVector).where(ChunkVector.chunk_id.in_([p.id for p in points])))
                session.add_all(
                    ChunkVector(
                        chunk_id=p.id, document_id=str(p.payload.get("document_id", "")), dimension=len(p.vector),
                        vector=np.asarray(p.vector, dtype="<f4").tobytes(), payload=p.payload,
                    )
                    for p in points
                )
        except Exception as exc:
            raise VectorStoreError("The vector index could not be written to the database.") from exc

    def search(self, vector: list[float], limit: int, flt: SearchFilter | None = None) -> list[VectorHit]:
        return self._refresh().search(vector, limit, flt)

    def delete_document(self, document_id: str) -> None:
        with session_scope(self._session_factory) as session:
            session.execute(delete(ChunkVector).where(ChunkVector.document_id == document_id))

    def count(self) -> int:
        with self._session_factory() as session:
            return int(session.execute(select(func.count(ChunkVector.chunk_id))).scalar_one())


def build_vector_store(settings: Settings, session_factory: SessionFactory | None = None) -> VectorStore:
    backend = settings.resolved_vector_store
    if backend == "database":
        if session_factory is None:
            raise VectorStoreError("VECTOR_STORE=database needs a database session factory.")
        return SqlVectorStore(session_factory)
    if backend == "memory":
        return LocalVectorStore(None)
    if backend == "local":
        return LocalVectorStore(settings.local_vector_path)
    if not settings.qdrant_url:
        raise VectorStoreError("VECTOR_STORE=qdrant requires QDRANT_URL.")
    return QdrantVectorStore(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection)
