"""Upload validation, path-safety and API-key authentication primitives."""
from __future__ import annotations

import hmac
import re
import zipfile
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from app.config import ALLOWED_EXTENSIONS, Settings
from app.utils.errors import (
    AuthenticationError,
    InvalidDocumentError,
    PermissionDeniedError,
    UnsupportedFormatError,
    ValidationAppError,
)

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ ()\-]+")


class Role(IntEnum):
    VIEWER = 1    # read-only: list, chat, analyse
    ANALYST = 2   # + upload documents, generate reports
    ADMIN = 3     # + delete documents, view traces


@dataclass(frozen=True)
class Principal:
    subject: str
    role: Role


ANONYMOUS_ADMIN = Principal(subject="anonymous", role=Role.ADMIN)


def parse_api_keys(raw: str) -> dict[str, Role]:
    """Parse ``key:role,key:role`` (role defaults to analyst)."""
    keys: dict[str, Role] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, _, role = entry.partition(":")
        try:
            keys[key.strip()] = Role[role.strip().upper()] if role.strip() else Role.ANALYST
        except KeyError as exc:
            raise ValueError(f"Unknown role in API_KEYS: {role!r}") from exc
    return keys


def authenticate(settings: Settings, presented_key: str | None) -> Principal:
    """Resolve the caller. With auth disabled every caller is a local admin."""
    if not settings.auth_enabled:
        return ANONYMOUS_ADMIN
    if not presented_key:
        raise AuthenticationError("An API key is required. Send it in the X-API-Key header.")
    for key, role in parse_api_keys(settings.api_keys).items():
        if hmac.compare_digest(key.encode(), presented_key.encode()):
            return Principal(subject=f"key:{key[:4]}…", role=role)
    raise AuthenticationError("The supplied API key is not valid.")


def require_role(principal: Principal, minimum: Role) -> None:
    if principal.role < minimum:
        raise PermissionDeniedError(f"This action requires the '{minimum.name.lower()}' role.")


# ── File handling ────────────────────────────────────────────────────────────
def sanitize_filename(name: str) -> str:
    """Return a display-safe filename. Never used to build a storage path."""
    name = Path(name.replace("\\", "/")).name
    name = _UNSAFE_CHARS.sub("_", name).strip(" .")
    return name[:180] or "document"


def validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UnsupportedFormatError(f"Unsupported file type '{ext or 'unknown'}'. Allowed: {allowed}.")
    return ext


def validate_magic_bytes(ext: str, head: bytes, path: Path | None = None) -> None:
    """Check that the content matches the claimed extension."""
    if not head:
        raise InvalidDocumentError("The uploaded file is empty.")
    if ext == ".pdf":
        if b"%PDF-" not in head[:1024]:
            raise InvalidDocumentError("The file has a .pdf extension but is not a valid PDF.")
    elif ext in (".docx", ".xlsx"):
        if not head.startswith(b"PK\x03\x04"):
            raise InvalidDocumentError(f"The file has a {ext} extension but is not a valid Office document.")
        if path is not None:
            marker = "word/" if ext == ".docx" else "xl/"
            try:
                with zipfile.ZipFile(path) as archive:
                    if not any(n.startswith(marker) for n in archive.namelist()):
                        raise InvalidDocumentError(f"The file is not a valid {ext} document.")
            except zipfile.BadZipFile as exc:
                raise InvalidDocumentError(f"The {ext} file is corrupted.") from exc
    else:  # plain text
        if b"\x00" in head:
            raise InvalidDocumentError("The text file appears to contain binary data.")


def safe_join(base: Path, relative: str) -> Path:
    """Join and guarantee the result stays inside ``base`` (path-traversal guard)."""
    base_resolved = base.resolve()
    candidate = (base_resolved / relative).resolve()
    if base_resolved != candidate and base_resolved not in candidate.parents:
        raise ValidationAppError("Invalid file path.")
    return candidate
