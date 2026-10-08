"""LLM provider abstraction, security primitives, caching and log redaction."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from app.llm.base import LLMUsage
from app.llm.factory import NullLLM, build_llm
from app.llm.openai_compatible import OpenAICompatibleClient
from app.utils.cache import InMemoryCache, build_cache
from app.utils.errors import (
    AuthenticationError,
    InvalidDocumentError,
    LLMError,
    LLMNotConfiguredError,
    LLMTimeoutError,
    PermissionDeniedError,
    RateLimitError,
    UnsupportedFormatError,
    ValidationAppError,
)
from app.utils.logging import JsonFormatter, redact
from app.utils.observability import Trace
from app.utils.rate_limit import RateLimiter
from app.utils.security import (
    Role,
    authenticate,
    parse_api_keys,
    require_role,
    safe_join,
    sanitize_filename,
    validate_extension,
    validate_magic_bytes,
)
from tests.conftest import make_settings

MESSAGES = [{"role": "user", "content": "hello"}]


def _client(handler, **kwargs) -> OpenAICompatibleClient:  # noqa: ANN001
    return OpenAICompatibleClient(
        provider=kwargs.pop("provider", "groq"), base_url="https://llm.test/v1", api_key="gsk_test_key_123456",
        model="test-model", max_retries=kwargs.pop("max_retries", 2), transport=httpx.MockTransport(handler), **kwargs,
    )


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    async def instant(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(OpenAICompatibleClient, "_backoff", instant)


class TestOpenAICompatibleClient:
    async def test_complete_sends_openai_format(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["auth"] = request.headers["authorization"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "model": "test-model", "choices": [{"message": {"content": "hi there"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            })

        response = await _client(handler).complete(MESSAGES, json_mode=True, max_tokens=50)
        assert response.text == "hi there" and response.usage.total_tokens == 9
        assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer gsk_test_key_123456"
        assert seen["body"]["model"] == "test-model" and seen["body"]["messages"] == MESSAGES
        assert seen["body"]["max_tokens"] == 50 and seen["body"]["response_format"] == {"type": "json_object"}

    async def test_retries_transient_failures(self):
        attempts = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(503, text="overloaded")
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        assert (await _client(handler).complete(MESSAGES)).text == "ok"
        assert attempts["n"] == 3

    @pytest.mark.parametrize(
        ("status", "fragment"), [(401, "API key"), (404, "does not recognise model"), (429, "rate-limiting"), (500, "HTTP 500")]
    )
    async def test_errors_are_user_safe(self, status, fragment):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, text="internal detail: key gsk_test_key_123456")

        with pytest.raises(LLMError) as exc:
            await _client(handler, max_retries=0).complete(MESSAGES)
        assert fragment in exc.value.message and "gsk_" not in exc.value.message

    async def test_timeout(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow")

        with pytest.raises(LLMTimeoutError):
            await _client(handler, max_retries=1).complete(MESSAGES)

    async def test_malformed_response(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": True})

        with pytest.raises(LLMError, match="unexpected response"):
            await _client(handler).complete(MESSAGES)

    async def test_streaming_yields_deltas_and_usage(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
            lines = [
                'data: {"choices":[{"delta":{"role":"assistant"}}]}',
                'data: {"choices":[{"delta":{"content":"Revenue "}}]}',
                ": keep-alive comment",
                'data: {"choices":[{"delta":{"content":"grew."}}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3}}',
                "data: [DONE]",
            ]
            return httpx.Response(200, text="\n\n".join(lines) + "\n\n", headers={"content-type": "text/event-stream"})

        usage = LLMUsage()
        deltas = [d async for d in _client(handler).stream(MESSAGES, usage=usage)]
        assert deltas == ["Revenue ", "grew."]
        assert (usage.prompt_tokens, usage.completion_tokens) == (11, 3)

    async def test_streaming_estimates_usage_when_not_reported(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "stream_options" not in json.loads(request.content)     # not sent to local servers
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"one two three"}}]}\n\ndata: [DONE]\n\n')

        usage = LLMUsage()
        assert [d async for d in _client(handler, provider="local").stream(MESSAGES, usage=usage)] == ["one two three"]
        assert usage.completion_tokens == 3 and usage.prompt_tokens == 1

    async def test_streaming_error_status(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="nope")

        with pytest.raises(LLMError, match="API key"):
            [d async for d in _client(handler).stream(MESSAGES)]


class TestProviderFactory:
    @pytest.mark.parametrize(
        ("overrides", "provider", "base_url"),
        [
            ({"llm_api_key": "gsk_abc"}, "groq", "https://api.groq.com/openai/v1"),
            ({"llm_api_key": "sk-abc"}, "openai", "https://api.openai.com/v1"),
            ({"llm_provider": "local"}, "local", "http://localhost:11434/v1"),
            ({"llm_provider": "openai", "llm_api_key": "k", "llm_base_url": "https://gateway.example/v1/"}, "openai", "https://gateway.example/v1"),
            ({"llm_provider": "auto", "llm_base_url": "http://vllm:8000/v1"}, "local", "http://vllm:8000/v1"),
        ],
    )
    def test_provider_resolution(self, tmp_path: Path, overrides, provider, base_url):
        settings = make_settings(tmp_path, **{"llm_provider": "auto", **overrides})
        assert settings.resolved_llm_provider == provider and settings.resolved_llm_base_url == base_url
        llm = build_llm(settings)
        assert llm.available and llm.provider == provider

    async def test_unconfigured_llm_is_explicit(self, tmp_path: Path):
        llm = build_llm(make_settings(tmp_path, llm_provider="auto"))
        assert isinstance(llm, NullLLM) and not llm.available
        with pytest.raises(LLMNotConfiguredError):
            await llm.complete(MESSAGES)
        # A hosted provider without a key degrades instead of failing at request time.
        assert not build_llm(make_settings(tmp_path, llm_provider="groq")).available

    def test_database_url_normalisation(self, tmp_path: Path):
        assert make_settings(tmp_path).resolved_database_url.startswith("sqlite:///")
        pg = make_settings(tmp_path, database_url="postgres://u:p@host:5432/db")
        assert pg.resolved_database_url == "postgresql+psycopg://u:p@host:5432/db"


class TestFileSecurity:
    @pytest.mark.parametrize(
        ("raw", "safe"),
        [
            ("../../etc/passwd.pdf", "passwd.pdf"), ("..\\..\\windows\\evil.pdf", "evil.pdf"),
            ("report<script>.pdf", "report_script_.pdf"), ("  .hidden.pdf ", "hidden.pdf"),
            ("normal report (final).pdf", "normal report (final).pdf"), ("", "document"),
        ],
    )
    def test_sanitize_filename(self, raw, safe):
        assert sanitize_filename(raw) == safe

    def test_extension_allow_list(self):
        assert validate_extension("Report.PDF") == ".pdf"
        for name in ("malware.exe", "page.html", "archive.zip", "noextension"):
            with pytest.raises(UnsupportedFormatError):
                validate_extension(name)

    def test_magic_bytes(self, tmp_path: Path):
        validate_magic_bytes(".pdf", b"%PDF-1.7\n...")
        validate_magic_bytes(".txt", b"plain text")
        with pytest.raises(InvalidDocumentError):
            validate_magic_bytes(".pdf", b"MZ\x90\x00 this is an executable")
        with pytest.raises(InvalidDocumentError):
            validate_magic_bytes(".docx", b"not a zip")
        with pytest.raises(InvalidDocumentError):
            validate_magic_bytes(".txt", b"abc\x00def")
        with pytest.raises(InvalidDocumentError):
            validate_magic_bytes(".pdf", b"")

    def test_office_container_is_inspected(self, tmp_path: Path):
        import zipfile

        fake = tmp_path / "fake.docx"
        with zipfile.ZipFile(fake, "w") as archive:
            archive.writestr("readme.txt", "just a zip")
        with pytest.raises(InvalidDocumentError):
            validate_magic_bytes(".docx", fake.read_bytes()[:2048], fake)

    def test_safe_join_blocks_traversal(self, tmp_path: Path):
        assert safe_join(tmp_path, "abc.pdf") == (tmp_path / "abc.pdf").resolve()
        for attempt in ("../outside.pdf", "a/../../outside.pdf", "C:/Windows/system32/x.pdf", "/etc/passwd"):
            with pytest.raises(ValidationAppError):
                safe_join(tmp_path, attempt)


class TestAuth:
    def test_disabled_auth_is_local_admin(self, tmp_path: Path):
        principal = authenticate(make_settings(tmp_path), None)
        assert principal.role == Role.ADMIN and principal.subject == "anonymous"

    def test_api_keys_and_roles(self, tmp_path: Path):
        settings = make_settings(tmp_path, auth_enabled=True, api_keys="view-key:viewer, work-key , root-key:admin")
        assert parse_api_keys(settings.api_keys) == {"view-key": Role.VIEWER, "work-key": Role.ANALYST, "root-key": Role.ADMIN}
        assert authenticate(settings, "view-key").role == Role.VIEWER
        assert "view-key" not in authenticate(settings, "view-key").subject        # never echo the full key
        with pytest.raises(AuthenticationError):
            authenticate(settings, None)
        with pytest.raises(AuthenticationError):
            authenticate(settings, "wrong")

    def test_role_hierarchy(self, tmp_path: Path):
        settings = make_settings(tmp_path, auth_enabled=True, api_keys="v:viewer,a:admin")
        require_role(authenticate(settings, "a"), Role.ADMIN)
        require_role(authenticate(settings, "v"), Role.VIEWER)
        with pytest.raises(PermissionDeniedError):
            require_role(authenticate(settings, "v"), Role.ANALYST)

    def test_bad_role_is_a_config_error(self):
        with pytest.raises(ValueError):
            parse_api_keys("key:superuser")


class TestCacheAndRateLimit:
    def test_in_memory_cache(self):
        cache = InMemoryCache(max_items=2)
        cache.set("a", [1, 2], ttl_seconds=60)
        assert cache.get("a") == [1, 2] and cache.get("missing") is None
        cache.set("expired", 1, ttl_seconds=-1)
        assert cache.get("expired") is None
        assert cache.incr("counter", 60) == 1 and cache.incr("counter", 60) == 2

    def test_unreachable_redis_falls_back(self):
        assert build_cache("redis://127.0.0.1:1/0").backend == "memory"
        assert build_cache("").backend == "memory"

    def test_rate_limiter(self):
        limiter = RateLimiter(InMemoryCache(), limit_per_minute=3)
        for _ in range(3):
            limiter.check("client-1")
        with pytest.raises(RateLimitError):
            limiter.check("client-1")
        limiter.check("client-2")                                   # limits are per client
        RateLimiter(InMemoryCache(), limit_per_minute=1, enabled=False).check("x")


class TestObservability:
    def test_secrets_are_redacted(self):
        # The key-type prefix is kept (useful when debugging); the secret part is masked.
        assert redact("calling with gsk_abcdefghijklmnop") == "calling with gsk_***"
        assert redact("Authorization: Bearer abcdefghijklmnop") == "Authorization: Bearer ***"
        assert redact("postgresql://user:hunter2@db:5432/x") == "postgresql://user:***@db:5432/x"
        assert redact("api_key=supersecretvalue next") == "api_key=*** next"

    def test_json_log_records_mask_sensitive_fields(self):
        record = logging.LogRecord("app", logging.INFO, __file__, 1, "connected using sk-abcdefghijklmnop", (), None)
        record.api_key = "sk-abcdefghijklmnop"
        record.documents = 3
        payload = json.loads(JsonFormatter().format(record))
        assert payload["api_key"] == "***" and payload["documents"] == 3
        assert "sk-abc" not in payload["msg"] and payload["level"] == "INFO"

    def test_trace_hashes_queries_by_default(self):
        trace = Trace("What is Aurora's confidential revenue?")
        with trace.span("retrieval"):
            pass
        trace.add_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        trace.set(intent="DOCUMENT_QA")
        summary = trace.summary()
        assert "confidential" not in json.dumps(summary)
        assert summary["query_chars"] == 38 and len(summary["query_hash"]) == 16
        assert summary["usage"]["total_tokens"] == 15 and "retrieval" in summary["timings_ms"]
        assert Trace("visible question", store_query_text=True).query_preview == "visible question"
