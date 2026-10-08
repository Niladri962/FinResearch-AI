from __future__ import annotations

from typing import AsyncIterator

from app.config import Settings
from app.llm.base import LLMClient, LLMResponse, LLMUsage, Message
from app.llm.openai_compatible import OpenAICompatibleClient
from app.utils.errors import LLMNotConfiguredError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class NullLLM(LLMClient):
    """Stands in when no provider is configured. Callers check ``available`` and
    fall back to extractive answers; calling it directly is a configuration error."""

    provider = "none"
    model = ""

    @property
    def available(self) -> bool:
        return False

    async def complete(self, messages: list[Message], **_: object) -> LLMResponse:
        raise LLMNotConfiguredError("No LLM provider is configured. Set LLM_API_KEY (or LLM_BASE_URL for a local model).")

    async def stream(self, messages: list[Message], *, usage: LLMUsage | None = None, **_: object) -> AsyncIterator[str]:
        raise LLMNotConfiguredError("No LLM provider is configured. Set LLM_API_KEY (or LLM_BASE_URL for a local model).")
        yield ""  # pragma: no cover - makes this an async generator


def build_llm(settings: Settings) -> LLMClient:
    provider = settings.resolved_llm_provider
    if provider == "none":
        logger.info("No LLM configured; answers will be extractive")
        return NullLLM()
    if provider in ("groq", "openai") and not settings.llm_api_key:
        logger.warning("LLM provider selected but LLM_API_KEY is missing; answers will be extractive")
        return NullLLM()
    return OpenAICompatibleClient(
        provider=provider,
        base_url=settings.resolved_llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
