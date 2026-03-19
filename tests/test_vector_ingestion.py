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
