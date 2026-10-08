"""Shared fixtures.

Everything runs offline: hash embeddings, in-memory vectors, lexical reranker,
SQLite in a temp directory and either no LLM or a scripted fake. Sample filings
are synthetic and generated on the fly (see scripts/generate_sample_data.py).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_sample_data import generate  # noqa: E402

from app.config import Settings  # noqa: E402
from app.llm.base import LLMClient, LLMResponse, LLMUsage, Message  # noqa: E402
from app.main import create_app  # noqa: E402


def make_settings(data_dir: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        environment="test", data_dir=data_dir, embedding_provider="hash", reranker_provider="lexical",
        vector_store="memory", llm_provider="none", llm_api_key="", redis_url="", database_url="",
        qdrant_url="", rate_limit_enabled=False, auth_enabled=False, log_level="WARNING",
        langchain_tracing_v2=False, otel_exporter_otlp_endpoint="",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


class FakeLLM(LLMClient):
    """Scripted model: ``responder(messages) -> str``. Records every call."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, responder: Callable[[list[Message]], str]) -> None:
        self.responder = responder
        self.calls: list[list[Message]] = []

    async def complete(self, messages: list[Message], **_: Any) -> LLMResponse:
        self.calls.append(messages)
        text = self.responder(messages)
        return LLMResponse(text=text, usage=LLMUsage(prompt_tokens=100, completion_tokens=20), model=self.model)

    async def stream(self, messages: list[Message], *, usage: LLMUsage | None = None, **_: Any) -> AsyncIterator[str]:
        self.calls.append(messages)
        text = self.responder(messages)
        for start in range(0, len(text), 24):
            yield text[start:start + 24]
        if usage is not None:
            usage.prompt_tokens, usage.completion_tokens = 100, 20


@pytest.fixture(scope="session")
def sample_docs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("samples")
    paths = generate(out, out / "dataset.jsonl")
    docs = {p.name: p for p in paths}
    docs["dataset"] = out / "dataset.jsonl"
    return docs


def upload(client: TestClient, path: Path, **form: Any) -> dict[str, Any]:
    with path.open("rb") as handle:
        response = client.post(
            "/api/documents/upload", params={"wait": "true"},
            files={"files": (path.name, handle, "application/octet-stream")}, data=form,
        )
    assert response.status_code == 202, response.text
    return response.json()["results"][0]["document"]


def ingest_samples(client: TestClient, sample_docs: dict[str, Path]) -> None:
    for name, path in sample_docs.items():
        if name == "dataset":
            continue
        document = upload(client, path)
        assert document["status"] == "ready", document


@pytest.fixture(scope="session")
def client(sample_docs: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """App with all sample filings ingested. Shared; tests must not delete its documents."""
    settings = make_settings(tmp_path_factory.mktemp("shared_data"))
    with TestClient(create_app(settings)) as test_client:
        ingest_samples(test_client, sample_docs)
        yield test_client


@pytest.fixture
def empty_client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(make_settings(tmp_path / "data"))) as test_client:
        yield test_client


@pytest.fixture
def company_ids(client: TestClient) -> dict[str, int]:
    return {c["name"].split()[0].lower(): c["id"] for c in client.get("/api/companies").json()}
