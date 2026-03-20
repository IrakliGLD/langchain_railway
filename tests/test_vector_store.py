import os

import pytest
import sqlalchemy

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


class _DummyResult:
    def fetchall(self):
        return []

    def keys(self):
        return []


class _DummyConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args, **kwargs):
        return _DummyResult()


class _DummyEngine:
    def connect(self):
        return _DummyConnection()


sqlalchemy.create_engine = lambda *args, **kwargs: _DummyEngine()  # type: ignore[assignment]

from contracts.vector_knowledge import VectorRetrievalFilters  # noqa: E402
from knowledge import vector_store  # noqa: E402


class _FakeMappingResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeConnection:
    def __init__(self, *, rows, captured):
        self._rows = rows
        self._captured = captured

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        self._captured["sql"] = str(sql)
        self._captured["params"] = dict(params)
        return _FakeMappingResult(self._rows)


class _FakeEngine:
    def __init__(self, *, rows, captured):
        self._rows = rows
        self._captured = captured

    def begin(self):
        return _FakeConnection(rows=self._rows, captured=self._captured)


def test_search_chunks_pushes_filters_into_sql(monkeypatch):
    captured = {}
    rows = [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "document_title": "Electricity Market Rules",
            "document_type": "regulation",
            "document_issuer": "GNERC",
            "source_key": "rules-1",
            "chunk_index": 0,
            "section_title": "Settlement",
            "section_path": "Settlement",
            "page_start": 2,
            "page_end": 2,
            "text_content": "Specific balancing settlement rule.",
            "token_count": 10,
            "language": "en",
            "topics": ["balancing_price"],
            "metadata": {"source_key": "rules-1"},
            "similarity_score": 0.91,
        }
    ]
    monkeypatch.setattr(vector_store, "ENGINE", _FakeEngine(rows=rows, captured=captured))
    store = vector_store.KnowledgeVectorStore()

    results = store.search_chunks(
        query_embedding=[0.1] * 1536,
        filters=VectorRetrievalFilters(
            preferred_topics=["balancing_price"],
            languages=["en", "ka"],
            document_types=["regulation"],
            issuers=["GNERC"],
        ),
        top_k=2,
        candidate_k=6,
    )

    sql = captured["sql"].lower()
    assert "jsonb_array_elements_text" in sql
    assert "c.language in" in sql
    assert "d.document_type in" in sql
    assert "d.issuer in" in sql
    assert results[0].document_issuer == "GNERC"


def test_search_chunks_rejects_wrong_embedding_dimension():
    store = vector_store.KnowledgeVectorStore()
    with pytest.raises(ValueError):
        store.search_chunks(query_embedding=[0.1, 0.2, 0.3])


def test_get_engine_disables_prepared_statements(monkeypatch):
    captured = {}

    def fake_create_engine(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _DummyEngine()

    vector_store._get_engine.cache_clear()
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setattr(vector_store, "create_engine", fake_create_engine)

    vector_store._get_engine()

    assert captured["kwargs"]["connect_args"]["prepare_threshold"] is None
