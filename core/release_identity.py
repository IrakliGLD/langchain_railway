"""Immutable build identity shared by runtime and release tooling."""

from __future__ import annotations

import json
import re
from pathlib import Path

RELEASE_IDENTITY_SCHEMA_VERSION = "backend-release-identity-v1"
DEFAULT_RELEASE_IDENTITY_PATH = Path("/app/release-identity.json")
_FULL_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_EXPECTED_KEYS = frozenset({"schema_version", "git_sha"})


def normalize_release_sha(value: str | None, *, field_name: str = "ENAI_RELEASE_SHA") -> str | None:
    """Return one normalized full Git SHA or fail closed for a supplied value."""
    normalized = (value or "").strip().lower()
    if not normalized:
        return None
    if not _FULL_GIT_SHA_RE.fullmatch(normalized):
        raise RuntimeError(f"{field_name} must be a full 40-character hexadecimal Git SHA")
    return normalized


def write_release_identity(path: Path, git_sha: str) -> None:
    """Write the deterministic build-time identity consumed by the runtime."""
    normalized = normalize_release_sha(git_sha, field_name="build release identity")
    if normalized is None:  # Defensive: the build identity is never optional.
        raise RuntimeError("build release identity must be a full 40-character hexadecimal Git SHA")
    payload = {
        "git_sha": normalized,
        "schema_version": RELEASE_IDENTITY_SCHEMA_VERSION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _read_embedded_release_sha(identity_path: Path) -> str | None:
    if not identity_path.exists():
        return None
    try:
        payload = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Embedded release identity is unreadable or invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _EXPECTED_KEYS:
        raise RuntimeError("Embedded release identity has an invalid field set")
    if payload.get("schema_version") != RELEASE_IDENTITY_SCHEMA_VERSION:
        raise RuntimeError("Embedded release identity has an unsupported schema version")
    return normalize_release_sha(
        payload.get("git_sha"),
        field_name="embedded release identity git_sha",
    )


def load_release_sha(
    *,
    identity_path: Path = DEFAULT_RELEASE_IDENTITY_PATH,
    environment_value: str | None = None,
) -> str | None:
    """Resolve the embedded image identity and reject a conflicting runtime claim."""
    embedded_sha = _read_embedded_release_sha(identity_path)
    environment_sha = normalize_release_sha(environment_value)
    if embedded_sha and environment_sha and embedded_sha != environment_sha:
        raise RuntimeError("ENAI_RELEASE_SHA does not match embedded image identity")
    return embedded_sha or environment_sha
