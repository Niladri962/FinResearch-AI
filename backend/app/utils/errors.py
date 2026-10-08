"""Domain exceptions.

Every error the API can surface derives from ``AppError`` and carries a stable
machine-readable ``code`` plus a user-safe message. Internal details (stack
traces, provider responses, secrets) never go into ``message``.
"""
from __future__ import annotations


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationAppError(AppError):
    status_code = 400
    code = "validation_error"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthenticated"


class PermissionDeniedError(AppError):
    status_code = 403
    code = "forbidden"


class RateLimitError(AppError):
    status_code = 429
    code = "rate_limited"


# ── Documents ────────────────────────────────────────────────────────────────
class UnsupportedFormatError(AppError):
    status_code = 415
    code = "unsupported_format"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class InvalidDocumentError(AppError):
    status_code = 422
    code = "invalid_document"


class EmptyDocumentError(AppError):
    status_code = 422
    code = "empty_document"


class OCRError(AppError):
    status_code = 422
    code = "ocr_failed"


# ── RAG infrastructure ───────────────────────────────────────────────────────
class EmbeddingError(AppError):
    status_code = 502
    code = "embedding_failed"


class VectorStoreError(AppError):
    status_code = 503
    code = "vector_store_unavailable"


class LLMError(AppError):
    status_code = 502
    code = "llm_error"


class LLMTimeoutError(LLMError):
    status_code = 504
    code = "llm_timeout"


class LLMNotConfiguredError(LLMError):
    status_code = 503
    code = "llm_not_configured"


# ── Financial engine ─────────────────────────────────────────────────────────
class MissingFinancialDataError(AppError):
    status_code = 404
    code = "missing_financial_data"


class CalculationError(AppError):
    status_code = 422
    code = "calculation_error"


class GuardrailViolation(AppError):
    status_code = 400
    code = "guardrail_violation"
