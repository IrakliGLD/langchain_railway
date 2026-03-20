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


def test_search_chunks_prefers_document_diversity_for_competitive_candidates(monkeypatch):
    captured = {}
    rows = [
        {
            "id": "chunk-1",
            "document_id": "doc-a",
            "document_title": "Doc A",
            "document_type": "regulation",
            "document_issuer": "Issuer A",
            "source_key": "doc-a",
            "chunk_index": 0,
            "section_title": "A1",
            "section_path": "A1",
            "page_start": 1,
            "page_end": 1,
            "text_content": "A1",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.93,
        },
        {
            "id": "chunk-2",
            "document_id": "doc-a",
            "document_title": "Doc A",
            "document_type": "regulation",
            "document_issuer": "Issuer A",
            "source_key": "doc-a",
            "chunk_index": 1,
            "section_title": "A2",
            "section_path": "A2",
            "page_start": 2,
            "page_end": 2,
            "text_content": "A2",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.91,
        },
        {
            "id": "chunk-3",
            "document_id": "doc-a",
            "document_title": "Doc A",
            "document_type": "regulation",
            "document_issuer": "Issuer A",
            "source_key": "doc-a",
            "chunk_index": 2,
            "section_title": "A3",
            "section_path": "A3",
            "page_start": 3,
            "page_end": 3,
            "text_content": "A3",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.90,
        },
        {
            "id": "chunk-4",
            "document_id": "doc-b",
            "document_title": "Doc B",
            "document_type": "regulation",
            "document_issuer": "Issuer B",
            "source_key": "doc-b",
            "chunk_index": 0,
            "section_title": "B1",
            "section_path": "B1",
            "page_start": 1,
            "page_end": 1,
            "text_content": "B1",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.89,
        },
    ]
    monkeypatch.setattr(vector_store, "ENGINE", _FakeEngine(rows=rows, captured=captured))
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT", 2)
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_DIVERSITY_SCORE_TOLERANCE", 0.08)
    store = vector_store.KnowledgeVectorStore()

    results = store.search_chunks(
        query_embedding=[0.1] * 1536,
        filters=VectorRetrievalFilters(preferred_topics=["market_structure"]),
        top_k=4,
        candidate_k=8,
    )

    assert [result.document_id for result in results] == ["doc-a", "doc-a", "doc-b", "doc-a"]


def test_search_chunks_keeps_single_document_when_other_candidates_are_weak(monkeypatch):
    captured = {}
    rows = [
        {
            "id": "chunk-1",
            "document_id": "doc-a",
            "document_title": "Doc A",
            "document_type": "regulation",
            "document_issuer": "Issuer A",
            "source_key": "doc-a",
            "chunk_index": 0,
            "section_title": "A1",
            "section_path": "A1",
            "page_start": 1,
            "page_end": 1,
            "text_content": "A1",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.95,
        },
        {
            "id": "chunk-2",
            "document_id": "doc-a",
            "document_title": "Doc A",
            "document_type": "regulation",
            "document_issuer": "Issuer A",
            "source_key": "doc-a",
            "chunk_index": 1,
            "section_title": "A2",
            "section_path": "A2",
            "page_start": 2,
            "page_end": 2,
            "text_content": "A2",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.94,
        },
        {
            "id": "chunk-3",
            "document_id": "doc-a",
            "document_title": "Doc A",
            "document_type": "regulation",
            "document_issuer": "Issuer A",
            "source_key": "doc-a",
            "chunk_index": 2,
            "section_title": "A3",
            "section_path": "A3",
            "page_start": 3,
            "page_end": 3,
            "text_content": "A3",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.93,
        },
        {
            "id": "chunk-4",
            "document_id": "doc-b",
            "document_title": "Doc B",
            "document_type": "regulation",
            "document_issuer": "Issuer B",
            "source_key": "doc-b",
            "chunk_index": 0,
            "section_title": "B1",
            "section_path": "B1",
            "page_start": 1,
            "page_end": 1,
            "text_content": "B1",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.30,
        },
    ]
    monkeypatch.setattr(vector_store, "ENGINE", _FakeEngine(rows=rows, captured=captured))
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT", 2)
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_DIVERSITY_SCORE_TOLERANCE", 0.08)
    store = vector_store.KnowledgeVectorStore()

    results = store.search_chunks(
        query_embedding=[0.1] * 1536,
        filters=VectorRetrievalFilters(preferred_topics=["market_structure"]),
        top_k=4,
        candidate_k=8,
    )

    assert [result.document_id for result in results] == ["doc-a", "doc-a", "doc-a"]


def test_search_chunks_reranks_capacity_market_title_match(monkeypatch):
    captured = {}
    rows = [
        {
            "id": "chunk-1",
            "document_id": "doc-day-ahead",
            "document_title": "Electricity Day-Ahead and Intraday Market Rules",
            "document_type": "regulation",
            "document_issuer": "GNERC",
            "source_key": "day-ahead-rules",
            "chunk_index": 0,
            "section_title": "Trading procedures",
            "section_path": "Trading procedures",
            "page_start": 1,
            "page_end": 1,
            "text_content": "Day-ahead trading procedures.",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.89,
        },
        {
            "id": "chunk-2",
            "document_id": "doc-capacity",
            "document_title": "Electricity Capacity Market Rules",
            "document_type": "regulation",
            "document_issuer": "GNERC",
            "source_key": "capacity-market-rules",
            "chunk_index": 0,
            "section_title": "Market participants",
            "section_path": "Market participants",
            "page_start": 1,
            "page_end": 1,
            "text_content": "Capacity market participants and obligations.",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.84,
        },
    ]
    monkeypatch.setattr(vector_store, "ENGINE", _FakeEngine(rows=rows, captured=captured))
    store = vector_store.KnowledgeVectorStore()

    results = store.search_chunks(
        query_embedding=[0.1] * 1536,
        filters=VectorRetrievalFilters(
            preferred_topics=["market_structure"],
            boost_terms=["capacity market", "capacity"],
        ),
        top_k=2,
        candidate_k=4,
    )

    assert results[0].document_id == "doc-capacity"


def test_search_chunks_reranks_export_section_match(monkeypatch):
    captured = {}
    rows = [
        {
            "id": "chunk-1",
            "document_id": "doc-day-ahead",
            "document_title": "Electricity Day-Ahead and Intraday Market Rules",
            "document_type": "regulation",
            "document_issuer": "GNERC",
            "source_key": "day-ahead-rules",
            "chunk_index": 0,
            "section_title": "General market operations",
            "section_path": "General market operations",
            "page_start": 1,
            "page_end": 1,
            "text_content": "General market operations.",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.87,
        },
        {
            "id": "chunk-2",
            "document_id": "doc-transitory",
            "document_title": "Transitory Market Rules",
            "document_type": "regulation",
            "document_issuer": "GNERC",
            "source_key": "transitory-market-rules",
            "chunk_index": 0,
            "section_title": "Export procedures and cross-border trade",
            "section_path": "Export procedures and cross-border trade",
            "page_start": 3,
            "page_end": 3,
            "text_content": "Electricity export requires defined cross-border procedures.",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.82,
        },
    ]
    monkeypatch.setattr(vector_store, "ENGINE", _FakeEngine(rows=rows, captured=captured))
    store = vector_store.KnowledgeVectorStore()

    results = store.search_chunks(
        query_embedding=[0.1] * 1536,
        filters=VectorRetrievalFilters(
            preferred_topics=["market_structure"],
            boost_terms=["export", "cross-border", "interconnection"],
        ),
        top_k=2,
        candidate_k=4,
    )

    assert results[0].document_id == "doc-transitory"


def test_search_chunks_reranks_balancing_buyers_match(monkeypatch):
    captured = {}
    rows = [
        {
            "id": "chunk-1",
            "document_id": "doc-day-ahead",
            "document_title": "Electricity Day-Ahead and Intraday Market Rules",
            "document_type": "regulation",
            "document_issuer": "GNERC",
            "source_key": "day-ahead-rules",
            "chunk_index": 0,
            "section_title": "Exchange registration",
            "section_path": "Exchange registration",
            "page_start": 1,
            "page_end": 1,
            "text_content": "Registration and exchange trading rules.",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.88,
        },
        {
            "id": "chunk-2",
            "document_id": "doc-capacity",
            "document_title": "Electricity Capacity Market Rules",
            "document_type": "regulation",
            "document_issuer": "GNERC",
            "source_key": "capacity-market-rules",
            "chunk_index": 0,
            "section_title": "Buyers of balancing electricity",
            "section_path": "Buyers of balancing electricity",
            "page_start": 2,
            "page_end": 2,
            "text_content": "The buyers of balancing electricity are defined in this section.",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.81,
        },
    ]
    monkeypatch.setattr(vector_store, "ENGINE", _FakeEngine(rows=rows, captured=captured))
    store = vector_store.KnowledgeVectorStore()

    results = store.search_chunks(
        query_embedding=[0.1] * 1536,
        filters=VectorRetrievalFilters(
            preferred_topics=["market_structure"],
            boost_terms=["balancing", "buyers", "buyer"],
        ),
        top_k=2,
        candidate_k=4,
    )

    assert results[0].document_id == "doc-capacity"
