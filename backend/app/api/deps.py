"""FastAPI dependencies: container access, authentication, RBAC and rate limiting."""
from __future__ import annotations

from typing import Annotated, Callable

from fastapi import Depends, Request

from app.services.container import AppContainer
from app.utils.security import Principal, Role, authenticate, require_role


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


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
