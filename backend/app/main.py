"""FastAPI application factory."""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import analysis, chat, companies, documents, health, reports
from app.config import Settings, get_settings
from app.llm.base import LLMClient
from app.services.container import AppContainer
from app.utils.errors import AppError
from app.utils.logging import configure_logging, get_logger
from app.utils.observability import configure_tracing

logger = get_logger("app")


def _error(status_code: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message, "details": details or {}}})


def create_app(settings: Settings | None = None, *, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        configure_tracing(settings)
        container = AppContainer(settings, llm=llm)
        recovered = container.documents.recover_interrupted()
        app.state.container = container
        logger.info(
            "startup",
            extra={
                "environment": settings.environment, "llm_provider": container.llm.provider,
                "embedding_provider": settings.embedding_provider, "vector_store": container.vector_store.backend,
                "database": container.engine.dialect.name, "recovered_documents": recovered,
            },
        )
        try:
            yield
        finally:
            await container.aclose()

    app = FastAPI(
        title=settings.app_name,
        description=settings.app_subtitle,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url=f"{settings.api_prefix}/docs",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_origin_regex=settings.cors_origin_regex or None,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # noqa: ANN001, ANN202
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        # Path only — query strings and bodies may contain user content and are never logged.
        logger.info(
            "request",
            extra={
                "request_id": request_id, "method": request.method, "path": request.url.path,
                "status": response.status_code, "ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        return _error(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        issues = [
            {"field": ".".join(str(p) for p in e.get("loc", []) if p != "body"), "message": e.get("msg", "")}
            for e in exc.errors()
        ]
        return _error(422, "validation_error", "The request is invalid.", {"issues": issues})

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Full details go to the log; the client gets a generic message with no internals.
        logger.exception("unhandled_error", extra={"path": request.url.path, "error": type(exc).__name__})
        return _error(500, "internal_error", "An unexpected error occurred. Please try again.")

    for module in (health, documents, companies, chat, analysis, reports):
        app.include_router(module.router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"name": settings.app_name, "docs": f"{settings.api_prefix}/docs", "health": f"{settings.api_prefix}/health"}

    return app


app = create_app()
