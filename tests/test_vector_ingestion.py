from contracts.vector_knowledge import DocumentRegistration
from knowledge.vector_chunking import chunk_markdown_text
from knowledge.vector_ingestion import VectorKnowledgeIngestor


class FakeEmbeddingProvider:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    def __init__(self):
        self.document = None
        self.saved_chunks = None
        self.saved_embeddings = None

    def upsert_document(self, document):
        self.document = document
        return "doc-123"

    def replace_document_chunks(self, *, document_id, source_key, chunks, embeddings):
        self.saved_chunks = chunks
        self.saved_embeddings = embeddings
        return {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "source_key": source_key,
        }


def test_chunk_markdown_text_is_heading_aware():
    chunks = chunk_markdown_text(
        "# Title\n\nParagraph one.\n\n## Section\n\nParagraph two.\n\nParagraph three.",
        language="en",
    )
    assert len(chunks) >= 2
    assert chunks[0].section_title == "Title"
    assert any(chunk.section_title == "Section" for chunk in chunks)


# ---------------------------------------------------------------------------
# Phase B.2 — chunker populates structural fields + outgoing refs
# ---------------------------------------------------------------------------


def test_chunker_extracts_article_number_from_georgian_heading():
    """``### მუხლი 14. Title`` produces a chunk with ``article_number=14``."""
    chunks = chunk_markdown_text(
        "### მუხლი 14. სისტემის კომერციული ოპერატორი\n\n"
        "ფასი დადგენილია ამ წესების შესაბამისად.",
        language="ka",
    )
    assert chunks
    assert chunks[0].article_number == "14"
    assert chunks[0].section_kind == "article"
    assert chunks[0].chapter_number == ""


def test_chunker_extracts_decimal_article_number_intact():
    """Decimal article ``### მუხლი 14.7`` keeps the ``14.7`` form."""
    chunks = chunk_markdown_text(
        "### მუხლი 14.7. Special case\n\nBody text here.",
        language="ka",
    )
    assert chunks
    assert chunks[0].article_number == "14.7"


def test_chunker_tracks_parent_chapter_ancestry():
    """An article that sits under a ``## თავი IV`` chapter heading must
    carry ``parent_chapter="IV"``. Pinned because the resolver uses this
    to answer "what chapter is article 14 in?" without a separate join."""
    chunks = chunk_markdown_text(
        "## თავი IV ელექტროენერგიის იმპორტი\n\n"
        "Introduction text.\n\n"
        "### მუხლი 14. სისტემის ოპერატორი\n\n"
        "Body text of article 14.\n\n"
        "### მუხლი 14.1. სპეციალური წესი\n\n"
        "Body text of article 14.1.",
        language="ka",
    )
    # Chapter heading chunk itself: parent_chapter is whatever came before
    # (empty here since IV is the first chapter).
    chapter_chunk = next(c for c in chunks if c.section_kind == "chapter")
    assert chapter_chunk.chapter_number == "IV"
    assert chapter_chunk.parent_chapter == ""

    # Both subsequent articles inherit chapter IV as parent.
    article_chunks = [c for c in chunks if c.section_kind == "article"]
    assert len(article_chunks) >= 2
    for chunk in article_chunks:
        assert chunk.parent_chapter == "IV"


def test_chunker_propagates_chapter_ancestry_across_empty_body_chapters():
    """Real Georgian regulations follow ``## თავი N <title>`` immediately
    with ``### მუხლი M`` — no body text between them. The chunker must
    still advance its chapter context through the empty-bodied chapter
    so the articles inherit the right ``parent_chapter``. Regression
    pin for the B.2.0 corpus-validation finding."""
    chunks = chunk_markdown_text(
        "## თავი I ზოგადი დებულებანი\n"
        "### მუხლი 1. მოქმედების სფერო\n\n"
        "Body of article 1.\n\n"
        "## თავი II საბითუმო ვაჭრობა\n"
        "### მუხლი 5. რეგისტრაცია\n\n"
        "Body of article 5.\n\n"
        "### მუხლი 6. დოკუმენტები\n\n"
        "Body of article 6.",
        language="ka",
    )
    articles = {c.article_number: c.parent_chapter for c in chunks if c.section_kind == "article"}
    assert articles == {"1": "I", "5": "II", "6": "II"}


def test_chunker_extracts_outgoing_refs_from_body_text():
    """The chunker parses references out of each chunk's body and
    attaches them as ``outgoing_refs``. Pinned form: user-cited
    ``მე-14 მუხლის მე-7 პუნქტი``."""
    chunks = chunk_markdown_text(
        "### მუხლი 30. სიტუაცია\n\n"
        "ვრცელდება ამ კანონის მე-14 მუხლის მე-7 პუნქტის შესაბამისად.",
        language="ka",
    )
    assert chunks
    refs = chunks[0].outgoing_refs
    assert len(refs) >= 1
    # The reference resolves to article 14 paragraph 7.
    target = next(r for r in refs if r.number == "14")
    assert target.sub_kind == "paragraph"
    assert target.sub_number == "7"


def test_chunker_drops_kodeksi_external_code_refs():
    """A chunk citing the Civil Code's article 30 must NOT emit a
    ``ChunkReference`` for it — the resolver would otherwise
    false-positive-match against a same-numbered article in the citing
    document."""
    chunks = chunk_markdown_text(
        "### მუხლი 5. ვადების გამოთვლა\n\n"
        "საქართველოს შრომის კოდექსის 30-ე მუხლის პირველი ნაწილით "
        "გათვალისწინებული უქმე დღეები არ ჩაითვლება.",
        language="ka",
    )
    assert chunks
    # No article reference with number "30" should be present.
    assert all(r.number != "30" for r in chunks[0].outgoing_refs)


def test_chunker_preserves_non_article_chunks_unchanged():
    """Free-form section bodies (no recognised article/chapter heading)
    still produce valid chunks — all the new structural fields default
    to empty strings, ``outgoing_refs`` defaults to an empty list."""
    chunks = chunk_markdown_text(
        "# General Definitions\n\n"
        "This section discusses general principles without any cross-references.",
        language="en",
    )
    assert chunks
    chunk = chunks[0]
    assert chunk.article_number == ""
    assert chunk.chapter_number == ""
    assert chunk.parent_chapter == ""
    assert chunk.section_kind == ""
    assert chunk.outgoing_refs == []


def test_ingestor_normalizes_document_type_and_persists_chunks():
    store = FakeStore()
    ingestor = VectorKnowledgeIngestor(store=store, embedding_provider=FakeEmbeddingProvider())
    result = ingestor.ingest_text_document(
        document=DocumentRegistration(
            source_key="doc-1",
            title="Electricity Law",
            document_type="market rule",
            language="en",
        ),
        text_content="# Section\n\nRegulatory text here.\n\nMore text.",
        topics=["regulation"],
    )
    assert store.document.document_type == "regulation"
    assert store.saved_chunks
    assert len(store.saved_chunks) == len(store.saved_embeddings)
    assert result["chunk_count"] == len(store.saved_chunks)
