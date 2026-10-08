"""LLM provider abstraction.

Agents depend only on ``LLMClient``. Any provider that can turn a list of chat
messages into text (optionally streamed) can be plugged in via ``build_llm``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResponse:
    text: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    model: str = ""


class LLMClient(ABC):
    provider: str = "abstract"
    model: str = ""

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    async def complete(
        self, messages: list[Message], *, temperature: float | None = None,
        max_tokens: int | None = None, json_mode: bool = False,
    ) -> LLMResponse: ...

    @abstractmethod
    def stream(
        self, messages: list[Message], *, usage: LLMUsage | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas. If ``usage`` is given it is filled in when the stream ends."""

    async def aclose(self) -> None:  # noqa: B027
        return None
