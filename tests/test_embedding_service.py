"""Embedding backend configuration and capability admission contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from knowledge import embedding_service
from knowledge.vector_embeddings import embedding_index_identity
from utils.request_deadline import current_request_execution_scope


def test_gemini_backend_prefers_dedicated_embedding_credential(monkeypatch):
    monkeypatch.setenv("GEMINI_EMBEDDING_API_KEY", '"embedding-only-key"')
    monkeypatch.setenv("GOOGLE_API_KEY", "generative-key")
    monkeypatch.delenv("ALLOW_LEGACY_GOOGLE_EMBEDDING_KEY", raising=False)
    monkeypatch.delenv("VECTOR_KNOWLEDGE_EMBEDDING_API_MODE", raising=False)

    backend = embedding_service.resolve_gemini_embedding_backend()

    assert backend.api_key == "embedding-only-key"
    assert backend.credential_source == "GEMINI_EMBEDDING_API_KEY"
    assert backend.api_mode == embedding_service.EmbeddingApiMode.DEVELOPER


def test_gemini_backend_rejects_implicit_legacy_credential(monkeypatch):
    monkeypatch.delenv("GEMINI_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "legacy-key")
    monkeypatch.delenv("ALLOW_LEGACY_GOOGLE_EMBEDDING_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_EMBEDDING_API_KEY"):
        embedding_service.resolve_gemini_embedding_backend()


def test_gemini_backend_allows_legacy_credential_only_with_explicit_flag(monkeypatch):
    monkeypatch.delenv("GEMINI_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "legacy-key")
    monkeypatch.setenv("ALLOW_LEGACY_GOOGLE_EMBEDDING_KEY", "true")

    backend = embedding_service.resolve_gemini_embedding_backend()

    assert backend.api_key == "legacy-key"
    assert backend.credential_source == "GOOGLE_API_KEY"


def test_gemini_backend_rejects_non_developer_api_mode(monkeypatch):
    monkeypatch.setenv("GEMINI_EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_API_MODE", "vertex")

    with pytest.raises(
        RuntimeError,
        match="VECTOR_KNOWLEDGE_EMBEDDING_API_MODE",
    ):
        embedding_service.resolve_gemini_embedding_backend()


def test_capability_probe_performs_one_embedding_call_and_returns_safe_status():
    provider = SimpleNamespace(
        _provider_name="gemini",
        _model="gemini-embedding-001",
        _expected_dimension=3,
        _api_mode="developer",
        _task_profile="retrieval_document_query_v1",
        calls=0,
    )

    def embed_query(text):
        assert text == embedding_service.EMBEDDING_CAPABILITY_PROBE_TEXT
        provider.calls += 1
        return [0.1, 0.2, 0.3]

    provider.embed_query = embed_query

    status = embedding_service.probe_embedding_capability(
        provider_factory=lambda: provider,
    )

    assert provider.calls == 1
    assert status.available is True
    assert status.public_payload() == {
        "available": True,
        "provider": "gemini",
        "model": "gemini-embedding-001",
        "dimension": 3,
        "api_mode": "developer",
        "task_profile": "retrieval_document_query_v1",
        "index_identity": embedding_index_identity(provider),
        "failure_disposition": None,
        "failure_reason": None,
    }


def test_capability_probe_binds_a_short_provider_deadline(monkeypatch):
    observed_remaining_seconds = []
    provider = SimpleNamespace(
        _provider_name="gemini",
        _model="gemini-embedding-001",
        _expected_dimension=1,
        _api_mode="developer",
    )

    def embed_query(_text):
        scope = current_request_execution_scope()
        assert scope is not None
        assert scope.deadline is not None
        observed_remaining_seconds.append(scope.deadline.remaining_seconds())
        return [0.1]

    provider.embed_query = embed_query
    monkeypatch.setenv(
        "VECTOR_KNOWLEDGE_EMBEDDING_CAPABILITY_TIMEOUT_SECONDS",
        "8",
    )

    status = embedding_service.probe_embedding_capability(
        provider_factory=lambda: provider,
    )

    assert status.available is True
    assert 0 < observed_remaining_seconds[0] <= 8


def test_required_capability_fails_closed_without_leaking_provider_error():
    secret = "credential-that-must-not-escape"

    def fail():
        raise RuntimeError(f"upstream rejected {secret}")

    status = embedding_service.probe_embedding_capability(provider_factory=fail)

    assert status.available is False
    assert secret not in str(status.public_payload())
    with pytest.raises(
        embedding_service.EmbeddingCapabilityUnavailable,
        match="Embedding capability unavailable",
    ) as exc_info:
        embedding_service.require_embedding_capability(
            probe=lambda: status,
        )
    assert secret not in str(exc_info.value)
