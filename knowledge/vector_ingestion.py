"""High-level ingestion helpers for vector-backed knowledge documents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from contracts.vector_knowledge import DocumentRegistration, IngestionResult
from knowledge.vector_catalogs import normalize_document_type
from knowledge.vector_chunking import chunk_markdown_text

if TYPE_CHECKING:
    from knowledge.vector_embeddings import EmbeddingProvider
    from knowledge.vector_store import KnowledgeVectorStore


class VectorKnowledgeIngestor:
    """Chunk, embed, and persist source documents into the vector store."""

    def __init__(
        self,
        *,
        store: "KnowledgeVectorStore" | None = None,
        embedding_provider: "EmbeddingProvider" | None = None,
    ) -> None:
        if store is None:
            from knowledge.vector_store import KnowledgeVectorStore

            store = KnowledgeVectorStore()
        self.store = store
        if embedding_provider is None:
            from knowledge.vector_embeddings import get_embedding_provider

            embedding_provider = get_embedding_provider()
        self.embedding_provider = embedding_provider

    def ingest_text_document(
        self,
        *,
        document: DocumentRegistration,
        text_content: str,
        topics: list[str] | None = None,
    ) -> IngestionResult:
        normalized_document = document.model_copy(
            update={"document_type": normalize_document_type(document.document_type)}
        )
        document_id = self.store.upsert_document(normalized_document)
        chunks = chunk_markdown_text(
            text_content,
            language=normalized_document.language,
            topics=list(topics or []),
            metadata={
                "source_key": normalized_document.source_key,
                "issuer": normalized_document.issuer,
                "document_type": normalized_document.document_type,
            },
        )
        embeddings = self.embedding_provider.embed_documents([chunk.text_content for chunk in chunks])
        return self.store.replace_document_chunks(
            document_id=document_id,
            source_key=normalized_document.source_key,
            chunks=chunks,
            embeddings=embeddings,
        )
