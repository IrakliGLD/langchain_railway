import hashlib
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


def test_google_sdk_uses_auth_key_header_without_oauth_bearer(monkeypatch):
    from google import genai

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    client = genai.Client(api_key="AQ.test-auth-key", vertexai=False)
    try:
        headers = client._api_client._http_options.headers
        assert headers["x-goog-api-key"] == "AQ.test-auth-key"
        assert "Authorization" not in headers
    finally:
        client.close()


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


def test_get_embedding_provider_selects_gemini(monkeypatch, caplog):
    captured = {}

    class FakeClient:
        def __init__(self, *, api_key, vertexai):
            captured["api_key"] = api_key
            captured["vertexai"] = vertexai
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
    monkeypatch.setenv("GEMINI_EMBEDDING_API_KEY", '"test-google-key"')
    monkeypatch.setenv("GOOGLE_API_KEY", "different-generative-key")
    caplog.set_level("INFO", logger="Enai")
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
    assert captured["vertexai"] is False
    assert captured["output_dimensionality"] == 768
    expected_fingerprint = hashlib.sha256(
        b"test-google-key"
    ).hexdigest()[:12]
    init_log = next(
        record.message
        for record in caplog.records
        if record.message.startswith("embedding_provider_initialized ")
    )
    assert "provider=gemini" in init_log
    assert "credential_source=GEMINI_EMBEDDING_API_KEY" in init_log
    assert f"credential_fingerprint={expected_fingerprint}" in init_log
    assert "test-google-key" not in init_log


def test_gemini_provider_validates_embedding_dimensions(monkeypatch):
    class FakeClient:
        def __init__(self, *, api_key, vertexai):
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
    monkeypatch.setenv("GEMINI_EMBEDDING_API_KEY", "test-google-key")
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
        def __init__(self, *, api_key, vertexai):
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
    monkeypatch.setenv("GEMINI_EMBEDDING_API_KEY", "test-google-key")
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


def test_gemini_retrieval_profile_uses_asymmetric_task_types_and_title(
    monkeypatch,
):
    captured = []

    class FakeClient:
        def __init__(self, *, api_key, vertexai):
            self.models = self

        def embed_content(self, *, model, contents, config):
            captured.append(
                {
                    "contents": contents,
                    "task_type": config.task_type,
                    "title": config.title,
                }
            )
            count = 1 if isinstance(contents, str) else len(contents)
            return types.SimpleNamespace(
                embeddings=[
                    types.SimpleNamespace(values=[0.2] * 768)
                    for _ in range(count)
                ]
            )

    class FakeEmbedContentConfig:
        def __init__(
            self,
            *,
            output_dimensionality,
            task_type=None,
            title=None,
        ):
            self.output_dimensionality = output_dimensionality
            self.task_type = task_type
            self.title = title

        def model_copy(self, *, update):
            return FakeEmbedContentConfig(
                output_dimensionality=self.output_dimensionality,
                task_type=update.get("task_type", self.task_type),
                title=update.get("title", self.title),
            )

    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv(
        "VECTOR_KNOWLEDGE_EMBEDDING_MODEL",
        "gemini-embedding-001",
    )
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION", "768")
    monkeypatch.setenv(
        "VECTOR_KNOWLEDGE_EMBEDDING_TASK_PROFILE",
        "retrieval_document_query_v1",
    )
    monkeypatch.setenv("GEMINI_EMBEDDING_API_KEY", "test-google-key")
    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeClient
    genai_module.types = types.SimpleNamespace(
        EmbedContentConfig=FakeEmbedContentConfig
    )
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    provider = vector_embeddings.GeminiEmbeddingProvider()
    provider.embed_documents(
        ["Article 1", "Article 2"],
        title="Electricity Market Rules",
    )
    provider.embed_query("Who can trade?")

    assert captured == [
        {
            "contents": ["Article 1", "Article 2"],
            "task_type": "RETRIEVAL_DOCUMENT",
            "title": "Electricity Market Rules",
        },
        {
            "contents": "Who can trade?",
            "task_type": "RETRIEVAL_QUERY",
            "title": None,
        },
    ]


def test_embedding_index_identity_changes_with_task_profile():
    class FakeProvider:
        _provider_name = "gemini"
        _model = "gemini-embedding-001"
        _expected_dimension = 1536
        _normalization_version = "v1"
        _corpus_version = "v2"
        _task_profile = "legacy"

    provider = FakeProvider()
    legacy_identity = vector_embeddings.embedding_index_identity(provider)
    provider._task_profile = "retrieval_document_query_v1"
    typed_identity = vector_embeddings.embedding_index_identity(provider)

    assert legacy_identity == "legacy"
    assert typed_identity != legacy_identity
    assert len(typed_identity) == 27


def test_gemini_provider_requires_google_genai(monkeypatch):
    # The legacy google-generativeai fallback was removed (F10 B6 F1.1);
    # google-genai is a hard production dependency and its absence must fail
    # loudly rather than silently degrade to a legacy backend.
    monkeypatch.setenv("GEMINI_EMBEDDING_API_KEY", "some-google-key")

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


def test_embedding_provider_cache_rotates_when_credential_changes(monkeypatch):
    class FakeEmbeddings:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_API_KEY", "first-key")
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        types.SimpleNamespace(OpenAIEmbeddings=FakeEmbeddings),
    )

    first = vector_embeddings.get_embedding_provider()
    monkeypatch.setenv("OPENAI_API_KEY", "second-key")
    second = vector_embeddings.get_embedding_provider()

    assert second is not first


def test_gemini_provider_cache_rotates_with_dedicated_credential(monkeypatch):
    class FakeClient:
        def __init__(self, *, api_key, vertexai):
            self.models = self

    class FakeEmbedContentConfig:
        def __init__(self, *, output_dimensionality):
            self.output_dimensionality = output_dimensionality

    monkeypatch.setenv("VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv(
        "VECTOR_KNOWLEDGE_EMBEDDING_MODEL",
        "gemini-embedding-001",
    )
    monkeypatch.setenv("GEMINI_EMBEDDING_API_KEY", "first-key")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeClient
    genai_module.types = types.SimpleNamespace(
        EmbedContentConfig=FakeEmbedContentConfig
    )
    google_module = types.ModuleType("google")
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    first = vector_embeddings.get_embedding_provider()
    monkeypatch.setenv("GEMINI_EMBEDDING_API_KEY", "second-key")
    second = vector_embeddings.get_embedding_provider()

    assert second is not first


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
        provider._task_profile = "retrieval_document_query_v1"
        _embed_query_cached(provider, "same query")
        provider._provider_name = "gemini"
        _embed_query_cached(provider, "same query")
        assert provider.calls == 7
    finally:
        reset_query_embedding_cache()
