"""Structured JSON logging with secret redaction."""
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

_SECRET_PATTERNS = [
    re.compile(r"\b(gsk_|sk-|xai-|lsv2_)[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)((?:api[_-]?key|password|secret|token)\s*[=:]\s*)\S+"),
    re.compile(r"(?i)(://[^:/\s]+:)[^@\s]+(@)"),  # credentials embedded in URLs
]
_SENSITIVE_KEYS = ("api_key", "apikey", "authorization", "password", "secret", "token")
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


def redact(value: str) -> str:
    """Mask anything that looks like a credential."""
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            value = pattern.sub(r"\1***\2", value)
        elif pattern.groups == 1:
            value = pattern.sub(r"\1***", value)
        else:
            value = pattern.sub("***", value)
    return value


def _safe(key: str, value: Any) -> Any:
    if any(s in key.lower() for s in _SENSITIVE_KEYS):
        return "***"
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(key, v) for v in value]
    if isinstance(value, dict):
        return {k: _safe(str(k), v) for k, v in value.items()}
    return redact(str(value))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = _safe(key, value)
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "pdfminer", "fastembed", "qdrant_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
