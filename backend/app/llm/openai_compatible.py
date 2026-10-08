"""Client for any OpenAI-compatible ``/chat/completions`` endpoint.

Covers Groq, OpenAI, and local servers (Ollama, vLLM, LM Studio) with a single
implementation — switching provider is a matter of base URL, key and model name.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import httpx

from app.llm.base import LLMClient, LLMResponse, LLMUsage, Message
from app.utils.errors import LLMError, LLMTimeoutError
from app.utils.logging import get_logger
from app.utils.text import count_tokens

logger = get_logger(__name__)

_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1500,
        timeout: float = 60.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
        )

    def _body(self, messages: list[Message], temperature: float | None, max_tokens: int | None) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }

    def _error(self, status: int) -> LLMError:
        if status in (401, 403):
            return LLMError("The LLM provider rejected the API key. Check LLM_API_KEY.")
        if status == 404:
            return LLMError(f"The LLM provider does not recognise model '{self.model}'. Check LLM_MODEL and LLM_BASE_URL.")
        if status == 429:
            return LLMError("The LLM provider is rate-limiting requests. Please retry shortly.")
        return LLMError(f"The LLM provider returned HTTP {status}.")

    async def _backoff(self, attempt: int, response: httpx.Response | None = None) -> None:
        delay = min(2.0 ** attempt, 8.0)
        if response is not None:
            try:
                delay = min(float(response.headers.get("retry-after", delay)), 15.0)
            except ValueError:
                pass
        await asyncio.sleep(delay)

    async def complete(
        self, messages: list[Message], *, temperature: float | None = None,
        max_tokens: int | None = None, json_mode: bool = False,
    ) -> LLMResponse:
        body = self._body(messages, temperature, max_tokens)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post("/chat/completions", json=body)
            except httpx.TimeoutException as exc:
                if attempt < self._max_retries:
                    await self._backoff(attempt)
                    continue
                raise LLMTimeoutError("The LLM provider took too long to respond.") from exc
            except httpx.HTTPError as exc:
                if attempt < self._max_retries:
                    await self._backoff(attempt)
                    continue
                raise LLMError("Could not reach the LLM provider. Check LLM_BASE_URL and network access.") from exc
            if response.status_code in _RETRYABLE and attempt < self._max_retries:
                await self._backoff(attempt, response)
                continue
            if response.status_code >= 400:
                raise self._error(response.status_code)
            try:
                data = response.json()
                text = data["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise LLMError("The LLM provider returned an unexpected response.") from exc
            raw_usage = data.get("usage") or {}
            usage = LLMUsage(
                prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
                completion_tokens=int(raw_usage.get("completion_tokens") or count_tokens(text)),
            )
            return LLMResponse(text=text, usage=usage, model=data.get("model", self.model))
        raise LLMError("The LLM request failed after retries.")  # pragma: no cover

    async def stream(
        self, messages: list[Message], *, usage: LLMUsage | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        body = self._body(messages, temperature, max_tokens)
        body["stream"] = True
        if self.provider != "local":  # some local servers reject unknown request fields
            body["stream_options"] = {"include_usage": True}
        produced: list[str] = []
        reported = False
        for attempt in range(self._max_retries + 1):
            try:
                async with self._client.stream("POST", "/chat/completions", json=body) as response:
                    if response.status_code in _RETRYABLE and attempt < self._max_retries:
                        await response.aread()
                        await self._backoff(attempt, response)
                        continue
                    if response.status_code >= 400:
                        await response.aread()
                        raise self._error(response.status_code)
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            event = json.loads(payload)
                        except ValueError:
                            continue
                        raw_usage = event.get("usage") or (event.get("x_groq") or {}).get("usage")
                        if raw_usage and usage is not None:
                            usage.prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
                            usage.completion_tokens = int(raw_usage.get("completion_tokens") or 0)
                            reported = True
                        choices = event.get("choices") or []
                        delta = (choices[0].get("delta") or {}).get("content") if choices else None
                        if delta:
                            produced.append(delta)
                            yield delta
                break
            except httpx.TimeoutException as exc:
                # Never retry once tokens have been emitted — the caller would see duplicated text.
                if attempt < self._max_retries and not produced:
                    await self._backoff(attempt)
                    continue
                raise LLMTimeoutError("The LLM provider took too long to respond.") from exc
            except httpx.HTTPError as exc:
                if attempt < self._max_retries and not produced:
                    await self._backoff(attempt)
                    continue
                raise LLMError("The connection to the LLM provider was interrupted.") from exc
        if usage is not None and not reported:
            usage.prompt_tokens = sum(count_tokens(m["content"]) for m in messages)
            usage.completion_tokens = count_tokens("".join(produced))

    async def aclose(self) -> None:
        await self._client.aclose()
