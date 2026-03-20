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

    class FakeClient:
        def __init__(self, *, api_key):
            captured["api_key"] = api_key
            self.models = self

        def embed_content(self, *, model, contents, config):
            captured["model"] = model
            captured["output_dimensionality"] = config.output_dimensionality
            embeddings = [[0.2] * 768] if isinstance(contents, str) else [[0.2] * 768 for _ in contents]
            return types.SimpleNamespace(
                embeddings=[types.SimpleNamespace(values=value) for value in embeddings]
            )

    class FakeEmbedContentConfig:
        def __init__(self, *, output_dimensionality):
            self.output_dimensionality = output_dimensionality

    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION", "768")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeClient
    genai_module.types = types.SimpleNamespace(EmbedContentConfig=FakeEmbedContentConfig)
    google_module.genai = genai_module
    monkeypatch.setitem(
        sys.modules,
        "google",
        google_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "google.genai",
        genai_module,
    )

    provider = vector_embeddings.get_embedding_provider()
    provider.embed_query("hello")

    assert isinstance(provider, vector_embeddings.GeminiEmbeddingProvider)
    assert captured["model"] == "gemini-embedding-001"
    assert captured["api_key"] == "test-google-key"
    assert captured["output_dimensionality"] == 768


def test_gemini_provider_validates_embedding_dimensions(monkeypatch):
    class FakeClient:
        def __init__(self, *, api_key):
            self.models = self

        def embed_content(self, *, model, contents, config):
            embeddings = [[0.2] * 767] if isinstance(contents, str) else [[0.2] * 767 for _ in contents]
            return types.SimpleNamespace(
                embeddings=[types.SimpleNamespace(values=value) for value in embeddings]
            )

    class FakeEmbedContentConfig:
        def __init__(self, *, output_dimensionality):
            self.output_dimensionality = output_dimensionality

    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION", "768")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeClient
    genai_module.types = types.SimpleNamespace(EmbedContentConfig=FakeEmbedContentConfig)
    google_module.genai = genai_module
    monkeypatch.setitem(
        sys.modules,
        "google",
        google_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "google.genai",
        genai_module,
    )

    provider = vector_embeddings.GeminiEmbeddingProvider()

    with pytest.raises(RuntimeError):
        provider.embed_query("hello")


def test_gemini_provider_batches_document_embeddings(monkeypatch):
    captured_batches = []

    class FakeClient:
        def __init__(self, *, api_key):
            self.models = self

        def embed_content(self, *, model, contents, config):
            if isinstance(contents, str):
                batch = [contents]
            else:
                batch = list(contents)
            captured_batches.append(len(batch))
            return types.SimpleNamespace(
                embeddings=[types.SimpleNamespace(values=[0.2] * 768) for _ in batch]
            )

    class FakeEmbedContentConfig:
        def __init__(self, *, output_dimensionality):
            self.output_dimensionality = output_dimensionality

    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION", "768")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_BATCH_SIZE", "100")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeClient
    genai_module.types = types.SimpleNamespace(EmbedContentConfig=FakeEmbedContentConfig)
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    provider = vector_embeddings.GeminiEmbeddingProvider()
    embeddings = provider.embed_documents([f"text {idx}" for idx in range(205)])

    assert len(embeddings) == 205
    assert captured_batches == [100, 100, 5]
