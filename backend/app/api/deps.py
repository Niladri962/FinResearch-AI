"""FastAPI dependencies: container access, authentication, RBAC and rate limiting."""
from __future__ import annotations

import threading
from typing import Annotated, Callable

from fastapi import Depends, FastAPI, Request

from app.services.container import AppContainer
from app.utils.security import Principal, Role, authenticate, require_role


_container_lock = threading.Lock()


def ensure_container(app: FastAPI) -> AppContainer:
    """Return the app's container, building it on first use.

    Normally the lifespan handler builds it at start-up. Some serverless runtimes
    never send ASGI lifespan events, so the first request builds it instead.
    """
    container = getattr(app.state, "container", None)
    if container is None:
        with _container_lock:
            container = getattr(app.state, "container", None)
            if container is None:
                container = app.state.container_factory()
                app.state.container = container
    return container


def get_container(request: Request) -> AppContainer:
    return ensure_container(request.app)


ContainerDep = Annotated[AppContainer, Depends(get_container)]


def get_principal(request: Request, container: ContainerDep) -> Principal:
    """Authenticate the caller and apply the per-client rate limit.

    With ``AUTH_ENABLED=false`` (local development) every caller is an anonymous
    admin. Turning auth on requires no route changes.
    """
    key = request.headers.get("x-api-key")
    if not key:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        key = token.strip() if scheme.lower() == "bearer" else None
    principal = authenticate(container.settings, key)
    client_id = principal.subject if container.settings.auth_enabled else (request.client.host if request.client else "local")
    container.rate_limiter.check(client_id)
    return principal


PrincipalDep = Annotated[Principal, Depends(get_principal)]


def requires(minimum: Role) -> Callable[[Principal], Principal]:
    def dependency(principal: PrincipalDep) -> Principal:
        require_role(principal, minimum)
        return principal
    return dependency


ViewerDep = Annotated[Principal, Depends(requires(Role.VIEWER))]
AnalystDep = Annotated[Principal, Depends(requires(Role.ANALYST))]
AdminDep = Annotated[Principal, Depends(requires(Role.ADMIN))]
