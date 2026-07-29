"""Embedding backend configuration and process-level capability admission."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from utils.provider_attempts import (
    classify_provider_failure,
    extract_failure_reason,
)
from utils.request_deadline import (
    bind_request_execution_scope_snapshot,
    cap_request_deadline,
)

EMBEDDING_CAPABILITY_PROBE_TEXT = "embedding readiness probe"


class EmbeddingApiMode(str, Enum):
    """Supported Google Gen AI transport modes for vector embeddings."""

    DEVELOPER = "developer"


@dataclass(frozen=True)
class GeminiEmbeddingBackend:
    """Resolved, explicit authentication contract for Gemini embeddings."""

    api_key: str
    credential_source: str
    api_mode: EmbeddingApiMode


@dataclass(frozen=True)
class EmbeddingCapabilityStatus:
    """Secret-free result of one process-level embedding canary."""

    available: bool
    provider: str
    model: str
    dimension: int
    api_mode: str
    task_profile: str = "legacy"
    index_identity: str = "legacy"
    failure_disposition: str | None = None
    failure_reason: str | None = None

    def public_payload(self) -> dict[str, bool | int | str | None]:
        return {
            "available": self.available,
            "provider": self.provider,
            "model": self.model,
            "dimension": self.dimension,
            "api_mode": self.api_mode,
            "task_profile": self.task_profile,
            "index_identity": self.index_identity,
            "failure_disposition": self.failure_disposition,
            "failure_reason": self.failure_reason,
        }


class EmbeddingCapabilityUnavailable(RuntimeError):
    """Raised when a process that requires embeddings cannot safely start."""


def _read_secret_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        value = value[1:-1].strip()
    return value


def resolve_gemini_embedding_backend() -> GeminiEmbeddingBackend:
    """Resolve the dedicated Developer API credential, failing closed."""

    raw_mode = (
        os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_API_MODE", "developer")
        .strip()
        .lower()
        or "developer"
    )
    try:
        api_mode = EmbeddingApiMode(raw_mode)
    except ValueError as exc:
        raise RuntimeError(
            "VECTOR_KNOWLEDGE_EMBEDDING_API_MODE must be 'developer'"
        ) from exc

    dedicated_key = _read_secret_env("GEMINI_EMBEDDING_API_KEY")
    if dedicated_key:
        return GeminiEmbeddingBackend(
            api_key=dedicated_key,
            credential_source="GEMINI_EMBEDDING_API_KEY",
            api_mode=api_mode,
        )

    raise RuntimeError(
        "GEMINI_EMBEDDING_API_KEY is required for Gemini vector embeddings"
    )


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default)).strip()))
    except ValueError:
        return default


def _configured_capability_identity() -> tuple[str, str, int, str, str]:
    provider = (
        os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER", "openai").strip().lower()
        or "openai"
    )
    if provider == "google":
        provider = "gemini"
    default_model = (
        "gemini-embedding-001"
        if provider == "gemini"
        else "text-embedding-3-small"
    )
    model = (
        os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", default_model).strip()
        or default_model
    )
    api_mode = (
        os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_API_MODE", "developer")
        .strip()
        .lower()
        or "developer"
        if provider == "gemini"
        else "public"
    )
    return (
        provider,
        model,
        _int_env("VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION", 1536),
        api_mode,
        (
            os.getenv(
                "VECTOR_KNOWLEDGE_EMBEDDING_TASK_PROFILE",
                "legacy",
            ).strip().lower()
            or "legacy"
        ),
    )


def _provider_identity(
    provider: object,
) -> tuple[str, str, int, str, str]:
    configured = _configured_capability_identity()
    return (
        str(getattr(provider, "_provider_name", configured[0])),
        str(getattr(provider, "_model", configured[1])),
        int(getattr(provider, "_expected_dimension", configured[2])),
        str(getattr(provider, "_api_mode", configured[3])),
        str(getattr(provider, "_task_profile", configured[4])),
    )


def probe_embedding_capability(
    *,
    provider_factory: Callable[[], object] | None = None,
) -> EmbeddingCapabilityStatus:
    """Perform exactly one embedding request and return only safe diagnostics."""

    if provider_factory is None:
        from knowledge.vector_embeddings import get_embedding_provider

        provider_factory = get_embedding_provider

    provider: object | None = None
    try:
        timeout_seconds = min(
            60,
            max(
                3,
                _int_env(
                    "VECTOR_KNOWLEDGE_EMBEDDING_CAPABILITY_TIMEOUT_SECONDS",
                    15,
                ),
            ),
        )
        deadline = cap_request_deadline(
            maximum_seconds=timeout_seconds,
            source="embedding_capability",
        )
        with bind_request_execution_scope_snapshot(deadline=deadline):
            provider = provider_factory()
            (
                provider_name,
                model,
                dimension,
                api_mode,
                task_profile,
            ) = _provider_identity(provider)
            from knowledge.vector_embeddings import (
                embedding_index_identity,
            )

            index_identity = embedding_index_identity(provider)
            embedding = provider.embed_query(  # type: ignore[attr-defined]
                EMBEDDING_CAPABILITY_PROBE_TEXT
            )
        actual_dimension = len(embedding)
        if actual_dimension != dimension:
            raise RuntimeError(
                "Embedding capability probe returned an incompatible dimension"
            )
        return EmbeddingCapabilityStatus(
            available=True,
            provider=provider_name,
            model=model,
            dimension=dimension,
            api_mode=api_mode,
            task_profile=task_profile,
            index_identity=index_identity,
        )
    except Exception as error:
        provider_name, model, dimension, api_mode, task_profile = (
            _provider_identity(provider)
            if provider is not None
            else _configured_capability_identity()
        )
        disposition = classify_provider_failure(error)
        if provider is not None:
            from knowledge.vector_embeddings import (
                embedding_index_identity,
            )

            index_identity = embedding_index_identity(provider)
        else:
            index_identity = (
                "legacy"
                if task_profile == "legacy"
                else "unknown"
            )
        return EmbeddingCapabilityStatus(
            available=False,
            provider=provider_name,
            model=model,
            dimension=dimension,
            api_mode=api_mode,
            task_profile=task_profile,
            index_identity=index_identity,
            failure_disposition=disposition.value,
            failure_reason=extract_failure_reason(error) or type(error).__name__,
        )


def require_embedding_capability(
    *,
    probe: Callable[[], EmbeddingCapabilityStatus] = probe_embedding_capability,
) -> EmbeddingCapabilityStatus:
    """Fail before work admission when the configured embedding path is down."""

    status = probe()
    if status.available:
        return status
    raise EmbeddingCapabilityUnavailable(
        "Embedding capability unavailable: "
        f"provider={status.provider} "
        f"disposition={status.failure_disposition or 'unknown'} "
        f"reason={status.failure_reason or 'unknown'}"
    )
