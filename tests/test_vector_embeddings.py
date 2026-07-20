import sys
import types

import pytest

from knowledge import vector_embeddings


@pytest.fixture(autouse=True)
def _isolate_provider_cache():
    vector_embeddings.reset_embedding_provider_cache()
    yield
    vector_embeddings.reset_embedding_provider_cache()


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


def test_gemini_provider_requires_google_genai(monkeypatch):
    # The legacy google-generativeai fallback was removed (F10 B6 F1.1);
    # google-genai is a hard production dependency and its absence must fail
    # loudly rather than silently degrade to a legacy backend.
    monkeypatch.setenv("GOOGLE_API_KEY", "some-google-key")

    google_module = types.ModuleType("google")
    google_module.__path__ = []  # package marker, but without a genai submodule
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.delitem(sys.modules, "google.genai", raising=False)
    monkeypatch.delitem(sys.modules, "google.generativeai", raising=False)

    with pytest.raises(RuntimeError, match="google-genai"):
        vector_embeddings.GeminiEmbeddingProvider()


def test_get_embedding_provider_caches_instance_per_config(monkeypatch):
    class FakeEmbeddings:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "text-embedding-3-cache-test")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION", "1536")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        types.SimpleNamespace(OpenAIEmbeddings=FakeEmbeddings),
    )

    vector_embeddings.reset_embedding_provider_cache()
    try:
        first = vector_embeddings.get_embedding_provider()
        second = vector_embeddings.get_embedding_provider()
        assert first is second

        # A config change keys a fresh instance.
        monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "text-embedding-3-cache-test-b")
        third = vector_embeddings.get_embedding_provider()
        assert third is not first

        vector_embeddings.reset_embedding_provider_cache()
        fourth = vector_embeddings.get_embedding_provider()
        assert fourth is not third
    finally:
        vector_embeddings.reset_embedding_provider_cache()


def test_query_embedding_cache_identity_includes_all_vector_compatibility_fields(monkeypatch):
    from knowledge.vector_retrieval import _embed_query_cached, reset_query_embedding_cache

    class FakeProvider:
        _provider_name = "openai"
        _model = "model-a"
        _expected_dimension = 3
        _normalization_version = "norm-v1"
        _corpus_version = "corpus-v1"

        def __init__(self):
            self.calls = 0

        def embed_query(self, _text):
            self.calls += 1
            return [float(self.calls)] * self._expected_dimension

    monkeypatch.setenv("VECTOR_QUERY_EMBEDDING_CACHE_SIZE", "16")
    provider = FakeProvider()
    reset_query_embedding_cache()
    try:
        first = _embed_query_cached(provider, "same query")
        assert _embed_query_cached(provider, "same query") == first
        assert provider.calls == 1

        provider._model = "model-b"
        _embed_query_cached(provider, "same query")
        provider._expected_dimension = 4
        _embed_query_cached(provider, "same query")
        provider._normalization_version = "norm-v2"
        _embed_query_cached(provider, "same query")
        provider._corpus_version = "corpus-v2"
        _embed_query_cached(provider, "same query")
        provider._provider_name = "gemini"
        _embed_query_cached(provider, "same query")
        assert provider.calls == 6
    finally:
        reset_query_embedding_cache()
