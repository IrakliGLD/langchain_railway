import sys
import types

import pytest

from knowledge import vector_embeddings


def test_resolved_provider_accepts_gemini_aliases():
    assert vector_embeddings._resolved_provider("gemini") == "gemini"
    assert vector_embeddings._resolved_provider("google") == "gemini"
    assert vector_embeddings._resolved_provider("openai") == "openai"


def test_resolved_provider_rejects_unknown_value():
    with pytest.raises(RuntimeError):
        vector_embeddings._resolved_provider("anthropic")


def test_get_embedding_provider_selects_openai(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def embed_documents(self, texts):
            return [[0.1] * 1536 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 1536

    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION", "1536")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        types.SimpleNamespace(OpenAIEmbeddings=FakeEmbeddings),
    )

    provider = vector_embeddings.get_embedding_provider()

    assert isinstance(provider, vector_embeddings.OpenAIEmbeddingProvider)
    assert captured["model"] == "text-embedding-3-small"
    assert captured["api_key"] == "test-openai-key"
    assert captured["dimensions"] == 1536


def test_get_embedding_provider_selects_gemini(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def embed_documents(self, texts):
            return [[0.2] * 768 for _ in texts]

        def embed_query(self, text):
            return [0.2] * 768

    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "models/text-embedding-004")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION", "768")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setitem(
        sys.modules,
        "langchain_google_genai",
        types.SimpleNamespace(GoogleGenerativeAIEmbeddings=FakeEmbeddings),
    )

    provider = vector_embeddings.get_embedding_provider()

    assert isinstance(provider, vector_embeddings.GeminiEmbeddingProvider)
    assert captured["model"] == "models/text-embedding-004"
    assert captured["google_api_key"] == "test-google-key"


def test_gemini_provider_validates_embedding_dimensions(monkeypatch):
    class FakeEmbeddings:
        def __init__(self, **kwargs):
            pass

        def embed_documents(self, texts):
            return [[0.2] * 767 for _ in texts]

        def embed_query(self, text):
            return [0.2] * 767

    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "models/text-embedding-004")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION", "768")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setitem(
        sys.modules,
        "langchain_google_genai",
        types.SimpleNamespace(GoogleGenerativeAIEmbeddings=FakeEmbeddings),
    )

    provider = vector_embeddings.GeminiEmbeddingProvider()

    with pytest.raises(RuntimeError):
        provider.embed_query("hello")
