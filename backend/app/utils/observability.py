"""Per-request tracing.

A ``Trace`` records stage latencies, token usage and the ids of retrieved sources
for one query. It is persisted to the ``query_traces`` table and, when
OpenTelemetry is installed and configured, mirrored as OTel spans. LangSmith
picks up LangGraph runs automatically through the ``LANGCHAIN_*`` variables.

Privacy: document text is never recorded. Query text is stored only as a hash
and length unless ``TRACE_QUERY_TEXT=true``.
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import Settings
from app.utils.logging import get_logger
from app.utils.text import sha256_text, truncate

logger = get_logger(__name__)

_otel_tracer: Any = None


def configure_tracing(settings: Settings) -> None:
    """Wire optional exporters. Safe to call when the libraries are absent."""
    global _otel_tracer
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)
    if not settings.otel_exporter_otlp_endpoint:
        return
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "finresearch-ai"}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint))
        )
        otel_trace.set_tracer_provider(provider)
        _otel_tracer = otel_trace.get_tracer("finresearch")
        logger.info("OpenTelemetry tracing enabled")
    except ImportError:
        logger.warning("OTEL endpoint set but opentelemetry packages are not installed")


class Trace:
    def __init__(self, query: str, *, store_query_text: bool = False, conversation_id: str | None = None) -> None:
        self.id = uuid.uuid4().hex
        self.conversation_id = conversation_id
        self.query_hash = sha256_text(query)[:16]
        self.query_chars = len(query)
        self.query_preview = truncate(query, 200) if store_query_text else None
        self.started = time.perf_counter()
        self.timings_ms: dict[str, float] = {}
        self.attributes: dict[str, Any] = {}
        self.usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.status = "ok"
        self.error: str | None = None

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        otel_cm = _otel_tracer.start_as_current_span(name) if _otel_tracer else None
        if otel_cm:
            otel_cm.__enter__()
        try:
            yield
        except BaseException as exc:
            if otel_cm:
                otel_cm.__exit__(type(exc), exc, exc.__traceback__)
                otel_cm = None
            raise
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self.timings_ms[name] = round(self.timings_ms.get(name, 0.0) + elapsed, 1)
            if otel_cm:
                otel_cm.__exit__(None, None, None)

    def add_timing(self, name: str, ms: float) -> None:
        self.timings_ms[name] = round(self.timings_ms.get(name, 0.0) + ms, 1)

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)

    def add_usage(self, usage: dict[str, int] | None) -> None:
        for key, value in (usage or {}).items():
            if key in self.usage and isinstance(value, int):
                self.usage[key] += value

    def fail(self, exc: BaseException) -> None:
        self.status = "error"
        self.error = type(exc).__name__

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 1)

    def summary(self) -> dict[str, Any]:
        return {
            "trace_id": self.id,
            "query_hash": self.query_hash,
            "query_chars": self.query_chars,
            "status": self.status,
            "error": self.error,
            "total_ms": self.total_ms,
            "timings_ms": dict(self.timings_ms),
            "usage": dict(self.usage),
            **self.attributes,
        }

    def log(self) -> None:
        logger.info("query_trace", extra=self.summary())
