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


def _chunk(
    chunk_id: str,
    document_id: str,
    section_path: str,
    *,
    similarity: float,
    chunk_index: int = 0,
    section_title: str = "",
):
    return vector_store.VectorChunkRecord(
        id=chunk_id,
        document_id=document_id,
        document_title=f"Doc {document_id}",
        document_type="regulation",
        document_issuer=f"Issuer {document_id}",
        source_key=document_id,
        chunk_index=chunk_index,
        section_title=section_title,
        section_path=section_path,
        page_start=1,
        page_end=1,
        text_content=chunk_id,
        token_count=10,
        language="ka",
        topics=["market_structure"],
        metadata={},
        similarity_score=similarity,
    )


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
    # preferred_topics are now a soft scoring boost, NOT a hard SQL filter
    assert "jsonb_array_elements_text" not in sql
    assert "c.language in" in sql
    assert "d.document_type in" in sql
    assert "d.issuer in" in sql
    assert results[0].document_issuer == "GNERC"


def test_search_chunks_rejects_wrong_embedding_dimension():
    store = vector_store.KnowledgeVectorStore()
    with pytest.raises(ValueError):
        store.search_chunks(query_embedding=[0.1, 0.2, 0.3])


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


def test_apply_document_diversity_caps_same_section_in_competitive_selection(monkeypatch):
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT", 3)
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_SECTION", 1)
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_DIVERSITY_SCORE_TOLERANCE", 0.08)

    candidates = [
        _chunk("chunk-1", "doc-a", "Section A", similarity=0.95, chunk_index=0, section_title="Section A"),
        _chunk("chunk-2", "doc-a", "Section A", similarity=0.94, chunk_index=1, section_title="Section A"),
        _chunk("chunk-3", "doc-a", "Section B", similarity=0.93, chunk_index=2, section_title="Section B"),
        _chunk("chunk-4", "doc-b", "Section C", similarity=0.92, chunk_index=0, section_title="Section C"),
    ]

    results = vector_store._apply_document_diversity(
        candidates,
        top_k=4,
        filters=VectorRetrievalFilters(preferred_topics=["market_structure"]),
    )

    assert [result.id for result in results[:3]] == ["chunk-1", "chunk-3", "chunk-4"]
    assert results[3].id == "chunk-2"


def test_apply_document_diversity_allows_multiple_sections_in_dominant_document(monkeypatch):
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT", 2)
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_SECTION", 1)
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_DIVERSITY_SCORE_TOLERANCE", 0.08)

    candidates = [
        _chunk("chunk-1", "doc-a", "Section A", similarity=0.95, chunk_index=0, section_title="Section A"),
        _chunk("chunk-2", "doc-a", "Section B", similarity=0.94, chunk_index=1, section_title="Section B"),
        _chunk("chunk-3", "doc-a", "Section C", similarity=0.93, chunk_index=2, section_title="Section C"),
        _chunk("chunk-4", "doc-b", "Section D", similarity=0.30, chunk_index=0, section_title="Section D"),
    ]

    results = vector_store._apply_document_diversity(
        candidates,
        top_k=4,
        filters=VectorRetrievalFilters(preferred_topics=["market_structure"]),
    )

    assert [result.id for result in results] == ["chunk-1", "chunk-2", "chunk-3"]


def test_apply_document_diversity_missing_section_metadata_uses_unique_chunk_key(monkeypatch):
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT", 3)
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_SECTION", 1)
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_DIVERSITY_SCORE_TOLERANCE", 0.08)

    candidates = [
        _chunk("chunk-1", "doc-a", "", similarity=0.95, chunk_index=0),
        _chunk("chunk-2", "doc-a", "", similarity=0.94, chunk_index=1),
        _chunk("chunk-3", "doc-a", "", similarity=0.93, chunk_index=2),
    ]

    results = vector_store._apply_document_diversity(
        candidates,
        top_k=3,
        filters=VectorRetrievalFilters(preferred_topics=["market_structure"]),
    )

    assert [result.id for result in results] == ["chunk-1", "chunk-2", "chunk-3"]


def test_apply_document_diversity_backfill_prefers_unseen_sections_before_repeats(monkeypatch):
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT", 3)
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_SECTION", 1)
    monkeypatch.setattr(vector_store, "VECTOR_KNOWLEDGE_DIVERSITY_SCORE_TOLERANCE", 0.08)

    candidates = [
        _chunk("chunk-1", "doc-a", "Section A", similarity=0.95, chunk_index=0, section_title="Section A"),
        _chunk("chunk-2", "doc-a", "Section A", similarity=0.94, chunk_index=1, section_title="Section A"),
        _chunk("chunk-3", "doc-b", "Section B", similarity=0.93, chunk_index=0, section_title="Section B"),
        _chunk("chunk-4", "doc-a", "Section C", similarity=0.70, chunk_index=2, section_title="Section C"),
    ]

    results = vector_store._apply_document_diversity(
        candidates,
        top_k=4,
        filters=VectorRetrievalFilters(preferred_topics=["market_structure"]),
    )

    assert [result.id for result in results] == ["chunk-1", "chunk-3", "chunk-4", "chunk-2"]


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


def test_search_chunks_reranks_market_concept_reference_for_market_design_queries(monkeypatch):
    captured = {}
    rows = [
        {
            "id": "chunk-1",
            "document_id": "doc-transitory",
            "document_title": "General Market Rules",
            "document_type": "regulation",
            "document_issuer": "GNERC",
            "source_key": "general-market-rules",
            "chunk_index": 0,
            "section_title": "General market operation",
            "section_path": "General market operation",
            "page_start": 1,
            "page_end": 1,
            "text_content": "General market rules.",
            "token_count": 10,
            "language": "ka",
            "topics": ["market_structure"],
            "metadata": {},
            "similarity_score": 0.87,
        },
        {
            "id": "chunk-2",
            "document_id": "doc-rules",
            "document_title": "Electricity (Capacity) Market Rules",
            "document_type": "regulation",
            "document_issuer": "GNERC",
            "source_key": "capacity-market-rules",
            "chunk_index": 0,
            "section_title": "Scope of regulation",
            "section_path": "Scope of regulation",
            "page_start": 1,
            "page_end": 1,
            "text_content": "Trading is regulated by the approved market model concept.",
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
            preferred_topics=["market_design", "electricity_market_transitory_model"],
            boost_terms=["transitory"],
        ),
        top_k=2,
        candidate_k=4,
    )

    assert results[0].document_id == "doc-rules"


# ---------------------------------------------------------------------------
# Phase B.1 — cross-reference contract + row materialiser
# ---------------------------------------------------------------------------


def test_chunk_reference_validates_minimum_fields():
    """``ChunkReference`` is strict: empty number rejected, unknown kinds
    rejected. The parser (Phase B.2) is the only intended producer; the
    contract pin protects against silent corruption from other writers."""
    from contracts.vector_knowledge import ChunkReference, ChunkReferenceKind

    ok = ChunkReference(
        kind=ChunkReferenceKind.article,
        number="14",
        sub_kind="paragraph",
        sub_number="7",
        raw_text="მე-14 მუხლის მე-7 პუნქტი",
    )
    assert ok.kind == ChunkReferenceKind.article
    assert ok.number == "14"

    # Empty number is rejected.
    with pytest.raises(ValueError):
        ChunkReference(kind=ChunkReferenceKind.article, number="")

    # Unknown kind is rejected (pydantic enum coercion).
    with pytest.raises(ValueError):
        ChunkReference(kind="paragraph", number="7")  # type: ignore[arg-type]


def test_chunk_reference_serialises_to_json_safe_dict():
    """``outgoing_refs`` is stored as JSONB; the round-trip via
    ``model_dump(mode="json")`` must produce dicts the store can write
    with ``json.dumps``."""
    import json as _json

    from contracts.vector_knowledge import ChunkReference, ChunkReferenceKind

    ref = ChunkReference(
        kind=ChunkReferenceKind.article,
        number="14.7",
        raw_text="14.7 მუხლი",
    )
    dumped = ref.model_dump(mode="json")
    # Must serialise without raising.
    _json.dumps([dumped])
    assert dumped["kind"] == "article"
    assert dumped["number"] == "14.7"


def test_row_to_chunk_record_populates_new_fields():
    """The helper that materialises SELECT rows must surface the Phase B.1
    columns onto the record. Without this, downstream consumers (Phase B.3
    resolver) can't see article_number / outgoing_refs."""
    row = {
        "id": "c-1",
        "document_id": "doc-A",
        "document_title": "Electricity (Capacity) Market Rules",
        "document_type": "regulation",
        "document_issuer": "GNERC",
        "source_key": "ecmr",
        "chunk_index": 5,
        "section_title": "მუხლი 14",
        "section_path": "თავი IV / მუხლი 14",
        "article_number": "14",
        "chapter_number": "IV",
        "parent_chapter": "IV",
        "section_kind": "article",
        "outgoing_refs": [
            {"kind": "article", "number": "8.1", "raw_text": "მე-8.1 მუხლი"},
        ],
        "page_start": None,
        "page_end": None,
        "text_content": "Body of article 14.",
        "token_count": 12,
        "language": "ka",
        "topics": ["market_structure"],
        "metadata": {},
    }
    out = vector_store._row_to_chunk_record(row, similarity_score=0.87)

    assert out.article_number == "14"
    assert out.chapter_number == "IV"
    assert out.parent_chapter == "IV"
    assert out.section_kind == "article"
    assert len(out.outgoing_refs) == 1
    assert out.outgoing_refs[0].number == "8.1"
    assert out.similarity_score == 0.87


def test_row_to_chunk_record_tolerates_legacy_pre_migration_rows():
    """Rows produced before the B.1 ALTER TABLE landed will not have the new
    keys. The materialiser must produce a valid record with empty defaults
    — never raise — so existing deployments keep retrieving while the
    migration rolls out."""
    legacy_row = {
        "id": "c-legacy",
        "document_id": "doc-A",
        "document_title": "Old Doc",
        "document_type": "regulation",
        "document_issuer": "",
        "source_key": "old",
        "chunk_index": 0,
        "section_title": "Intro",
        "section_path": "Intro",
        # article_number, chapter_number, parent_chapter, section_kind,
        # outgoing_refs all absent.
        "page_start": None,
        "page_end": None,
        "text_content": "Pre-migration chunk text.",
        "token_count": 5,
        "language": "ka",
        "topics": [],
        "metadata": {},
    }
    out = vector_store._row_to_chunk_record(legacy_row, similarity_score=None)

    assert out.article_number == ""
    assert out.chapter_number == ""
    assert out.parent_chapter == ""
    assert out.section_kind == ""
    assert out.outgoing_refs == []
    assert out.similarity_score is None


def test_parse_outgoing_refs_handles_none_string_and_malformed():
    """The JSONB column may arrive as None (legacy), a JSON string (some
    drivers), or a malformed list (operator error in a manual update).
    Round-trip semantics must stay forgiving — drop bad entries, return
    the good ones."""
    from contracts.vector_knowledge import ChunkReferenceKind

    assert vector_store._parse_outgoing_refs(None) == []
    assert vector_store._parse_outgoing_refs([]) == []
    # JSON-string form (some drivers return this for jsonb columns).
    refs = vector_store._parse_outgoing_refs(
        '[{"kind": "article", "number": "14", "raw_text": "მუხლი 14"}]'
    )
    assert len(refs) == 1
    assert refs[0].kind == ChunkReferenceKind.article
    # Malformed entry mixed with a valid one — drop the bad, keep the good.
    refs = vector_store._parse_outgoing_refs(
        [
            {"this": "is", "garbage": True},  # missing required fields
            {"kind": "chapter", "number": "IV", "raw_text": "თავი IV"},
        ]
    )
    assert len(refs) == 1
    assert refs[0].kind == ChunkReferenceKind.chapter
    assert refs[0].number == "IV"


def test_replace_document_chunks_emits_new_columns_in_insert(monkeypatch):
    """The INSERT path must wire the new Phase B.1 columns through with the
    structural defaults filled in. Capturing the SQL statement and one set
    of params is enough — the rest is plain templating."""
    captured = {"sql": "", "params": {}}

    class _CapturingConnection(_FakeConnection):
        def execute(self, sql, params):
            # First call is the DELETE (no params dict for our purposes),
            # second is the first INSERT. We capture the INSERT.
            if "insert into" in str(sql).lower():
                captured["sql"] = str(sql)
                captured["params"] = dict(params)
            return _FakeMappingResult([])

    class _CapturingEngine(_FakeEngine):
        def begin(self):
            return _CapturingConnection(rows=self._rows, captured=self._captured)

    monkeypatch.setattr(vector_store, "ENGINE", _CapturingEngine(rows=[], captured=captured))

    store = vector_store.KnowledgeVectorStore()
    chunk = vector_store.ChunkIngestRecord(
        chunk_index=0,
        text_content="Article 14 body.",
        section_title="მუხლი 14",
        article_number="14",
        chapter_number="IV",
        parent_chapter="IV",
        section_kind="article",
        outgoing_refs=[],
    )
    store.replace_document_chunks(
        document_id="11111111-1111-1111-1111-111111111111",
        source_key="src",
        chunks=[chunk],
        embeddings=[[0.0] * 1536],
    )

    assert "article_number" in captured["sql"]
    assert "chapter_number" in captured["sql"]
    assert "outgoing_refs" in captured["sql"]
    assert captured["params"]["article_number"] == "14"
    assert captured["params"]["chapter_number"] == "IV"
    assert captured["params"]["section_kind"] == "article"
    # outgoing_refs serialised as a JSON string (cast(:outgoing_refs as jsonb)).
    assert captured["params"]["outgoing_refs"] == "[]"
