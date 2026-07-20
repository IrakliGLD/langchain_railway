"""Embedding providers for vector-backed knowledge retrieval."""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, List, Literal, Protocol

from utils.provider_attempts import (
    ProviderDeliveryDisposition,
    claim_provider_attempt,
    classify_provider_failure,
    finish_provider_attempt,
    wrap_provider_failure,
)
from utils.request_deadline import current_request_execution_scope


def _positive_float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except ValueError:
        return default
    return value if value > 0 else default


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _embedding_timeout_seconds(provider: str, stage: str) -> float:
    timeout_env = "GEMINI_TIMEOUT_SECONDS" if provider == "gemini" else "OPENAI_TIMEOUT_SECONDS"
    configured_seconds = _positive_float_env(timeout_env, 120.0)
    scope = current_request_execution_scope()
    if scope is None or scope.deadline is None:
        return configured_seconds
    return scope.deadline.bounded_timeout_seconds(
        f"provider_{provider}_{stage}",
        configured_timeout_seconds=configured_seconds,
        cleanup_allowance_ms=_bounded_int_env("ENAI_REQUEST_CLEANUP_ALLOWANCE_MS", 3000, 250, 15000),
        minimum_start_ms=_bounded_int_env("ENAI_PROVIDER_MINIMUM_START_BUDGET_MS", 500, 100, 5000),
    )


def _execute_embedding_call(provider: str, stage: str, call: Callable[[], Any]) -> Any:
    token = claim_provider_attempt(provider, stage)
    try:
        result = call()
    except Exception as error:
        disposition = classify_provider_failure(error)
        finish_provider_attempt(token, disposition)
        raise wrap_provider_failure(
            error,
            provider=provider,
            stage=stage,
            disposition=disposition,
        ) from error
    finish_provider_attempt(token, ProviderDeliveryDisposition.COMPLETED)
    return result


class EmbeddingProvider(Protocol):
    """Minimal embedding interface used by ingestion and retrieval."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


def _expected_dimension() -> int:
    raw = os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION", "1536").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 1536


def _batch_size() -> int:
    raw = os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_BATCH_SIZE", "100").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 100


def _resolved_provider(value: str | None = None) -> Literal["openai", "gemini"]:
    raw = (value or os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER", "openai")).strip().lower()
    if raw in {"google", "gemini"}:
        return "gemini"
    if raw in {"openai", ""}:
        return "openai"
    raise RuntimeError("Unsupported VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER. Expected 'openai' or 'gemini'.")


def _validate_embedding_dimensions(
    vectors: List[List[float]],
    *,
    expected_dimension: int,
    label: str,
) -> List[List[float]]:
    for idx, vector in enumerate(vectors):
        actual_dimension = len(vector)
        if actual_dimension != expected_dimension:
            raise RuntimeError(
                f"{label}[{idx}] dimension mismatch: expected {expected_dimension}, got {actual_dimension}"
            )
    return vectors


class OpenAIEmbeddingProvider:
    """OpenAI-backed embedding provider for 1536-dimensional vectors."""

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        resolved_model = (model or os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "text-embedding-3-small")).strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for vector embeddings")
        from langchain_openai import OpenAIEmbeddings

        self._expected_dimension = _expected_dimension()
        self._provider_name = "openai"
        self._model = resolved_model
        self._normalization_version = os.getenv("VECTOR_KNOWLEDGE_NORMALIZATION_VERSION", "v1").strip() or "v1"
        self._corpus_version = os.getenv("VECTOR_KNOWLEDGE_CORPUS_VERSION", "v1").strip() or "v1"
        client_kwargs = {
            "model": resolved_model,
            "api_key": api_key,
            "max_retries": 0,
            "request_timeout": _positive_float_env("OPENAI_TIMEOUT_SECONDS", 120.0),
        }
        if resolved_model.startswith("text-embedding-3"):
            client_kwargs["dimensions"] = self._expected_dimension
        try:
            self._client = OpenAIEmbeddings(**client_kwargs)
        except TypeError as exc:
            if "dimensions" in client_kwargs:
                raise RuntimeError(
                    "Configured embedding dimension enforcement is not supported by the installed langchain_openai version"
                ) from exc
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        timeout = _embedding_timeout_seconds("openai", "document_embedding")
        embeddings = _execute_embedding_call(
            "openai",
            "document_embedding",
            lambda: self._client.embed_documents(list(texts), timeout=timeout),
        )
        return _validate_embedding_dimensions(
            embeddings,
            expected_dimension=self._expected_dimension,
            label="document_embedding",
        )

    def embed_query(self, text: str) -> List[float]:
        timeout = _embedding_timeout_seconds("openai", "query_embedding")
        embedding = _execute_embedding_call(
            "openai",
            "query_embedding",
            lambda: self._client.embed_query(text, timeout=timeout),
        )
        return _validate_embedding_dimensions(
            [embedding],
            expected_dimension=self._expected_dimension,
            label="query_embedding",
        )[0]


class GeminiEmbeddingProvider:
    """Gemini-backed embedding provider using the available Google SDK."""

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        resolved_model = (model or os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "gemini-embedding-001")).strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini vector embeddings")

        self._expected_dimension = _expected_dimension()
        self._batch_size = _batch_size()
        self._provider_name = "gemini"
        self._model = resolved_model
        self._normalization_version = os.getenv("VECTOR_KNOWLEDGE_NORMALIZATION_VERSION", "v1").strip() or "v1"
        self._corpus_version = os.getenv("VECTOR_KNOWLEDGE_CORPUS_VERSION", "v1").strip() or "v1"
        # google-genai is a hard production dependency (the legacy
        # google-generativeai SDK was removed in F10 B2.A.5); there is no
        # fallback backend.
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini vector embeddings require google-genai"
            ) from exc
        self._types = types
        if hasattr(types, "HttpOptions") and hasattr(types, "HttpRetryOptions"):
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=max(
                        1,
                        int(_positive_float_env("GEMINI_TIMEOUT_SECONDS", 120.0) * 1000),
                    ),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        else:
            self._client = genai.Client(api_key=api_key)
        self._config = types.EmbedContentConfig(
            output_dimensionality=self._expected_dimension,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        timeout = _embedding_timeout_seconds("gemini", "document_embedding")

        def _call() -> List[List[float]]:
            embeddings: List[List[float]] = []
            items = list(texts)
            config = self._config
            if hasattr(self._types, "HttpOptions") and hasattr(self._types, "HttpRetryOptions"):
                config = self._config.model_copy(
                    update={
                        "http_options": self._types.HttpOptions(
                            timeout=max(1, int(timeout * 1000)),
                            retry_options=self._types.HttpRetryOptions(attempts=1),
                        )
                    }
                )
            for start in range(0, len(items), self._batch_size):
                batch = items[start : start + self._batch_size]
                result = self._client.models.embed_content(
                    model=self._model,
                    contents=batch,
                    config=config,
                )
                embeddings.extend(list(item.values) for item in (result.embeddings or []))
            return embeddings

        embeddings = _execute_embedding_call("gemini", "document_embedding", _call)
        return _validate_embedding_dimensions(
            embeddings,
            expected_dimension=self._expected_dimension,
            label="document_embedding",
        )

    def embed_query(self, text: str) -> List[float]:
        timeout = _embedding_timeout_seconds("gemini", "query_embedding")

        def _call() -> List[float]:
            config = self._config
            if hasattr(self._types, "HttpOptions") and hasattr(self._types, "HttpRetryOptions"):
                config = self._config.model_copy(
                    update={
                        "http_options": self._types.HttpOptions(
                            timeout=max(1, int(timeout * 1000)),
                            retry_options=self._types.HttpRetryOptions(attempts=1),
                        )
                    }
                )
            result = self._client.models.embed_content(
                model=self._model,
                contents=text,
                config=config,
            )
            embeddings = [list(item.values) for item in (result.embeddings or [])]
            if not embeddings:
                raise RuntimeError("Gemini embedding response did not contain any embeddings")
            return embeddings[0]

        embedding = _execute_embedding_call("gemini", "query_embedding", _call)
        return _validate_embedding_dimensions(
            [embedding],
            expected_dimension=self._expected_dimension,
            label="query_embedding",
        )[0]


# Provider construction builds an SDK client (and its HTTP session) — reuse it
# across requests instead of rebuilding per retrieval. Keyed by the env values
# that determine construction, so a config change never serves a stale client.
_PROVIDER_CACHE: dict[tuple[str, str, str, str, str, str], EmbeddingProvider] = {}
_PROVIDER_CACHE_LOCK = threading.Lock()


def _provider_cache_key(resolved_provider: str) -> tuple[str, str, str, str, str, str]:
    default_model = "gemini-embedding-001" if resolved_provider == "gemini" else "text-embedding-3-small"
    return (
        resolved_provider,
        os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", default_model).strip() or default_model,
        str(_expected_dimension()),
        os.getenv("VECTOR_KNOWLEDGE_NORMALIZATION_VERSION", "v1").strip() or "v1",
        os.getenv("VECTOR_KNOWLEDGE_CORPUS_VERSION", "v1").strip() or "v1",
        str(_batch_size()),
    )


def embedding_cache_identity(provider: EmbeddingProvider) -> tuple[str, str, int, str, str]:
    """Return every configuration dimension that makes vectors compatible."""
    return (
        str(getattr(provider, "_provider_name", type(provider).__name__)),
        str(getattr(provider, "_model", "")),
        int(getattr(provider, "_expected_dimension", _expected_dimension())),
        str(getattr(provider, "_normalization_version", "v1")),
        str(getattr(provider, "_corpus_version", "v1")),
    )


def reset_embedding_provider_cache() -> None:
    """Clear cached provider instances (tests / config reloads)."""
    with _PROVIDER_CACHE_LOCK:
        _PROVIDER_CACHE.clear()


def get_embedding_provider(provider: str | None = None) -> EmbeddingProvider:
    """Return the default embedding provider for vector knowledge.

    Instances are cached per provider/model/dimension/normalization/corpus
    identity (plus SDK batch size), so repeated retrievals reuse one SDK
    client without surviving a relevant configuration change.
    """

    resolved_provider = _resolved_provider(provider)
    key = _provider_cache_key(resolved_provider)
    with _PROVIDER_CACHE_LOCK:
        cached = _PROVIDER_CACHE.get(key)
    if cached is not None:
        return cached
    built: EmbeddingProvider
    if resolved_provider == "gemini":
        built = GeminiEmbeddingProvider()
    else:
        built = OpenAIEmbeddingProvider()
    with _PROVIDER_CACHE_LOCK:
        return _PROVIDER_CACHE.setdefault(key, built)
