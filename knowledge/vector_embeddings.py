"""Embedding providers for vector-backed knowledge retrieval."""

from __future__ import annotations

import os
from typing import List, Literal, Protocol


class EmbeddingProvider(Protocol):
    """Minimal embedding interface used by ingestion and retrieval."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


def _expected_dimension() -> int:
    raw = os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION", "1536").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 1536


def _batch_size() -> int:
    raw = os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_BATCH_SIZE", "100").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 100


def _resolved_provider(value: str | None = None) -> Literal["openai", "gemini"]:
    raw = (value or os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER", "openai")).strip().lower()
    if raw in {"google", "gemini"}:
        return "gemini"
    if raw in {"openai", ""}:
        return "openai"
    raise RuntimeError(
        "Unsupported VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER. Expected 'openai' or 'gemini'."
    )


def _validate_embedding_dimensions(
    vectors: List[List[float]],
    *,
    expected_dimension: int,
    label: str,
) -> List[List[float]]:
    for idx, vector in enumerate(vectors):
        actual_dimension = len(vector)
        if actual_dimension != expected_dimension:
            raise RuntimeError(
                f"{label}[{idx}] dimension mismatch: expected {expected_dimension}, got {actual_dimension}"
            )
    return vectors


class OpenAIEmbeddingProvider:
    """OpenAI-backed embedding provider for 1536-dimensional vectors."""

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        resolved_model = (model or os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "text-embedding-3-small")).strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for vector embeddings")
        from langchain_openai import OpenAIEmbeddings

        self._expected_dimension = _expected_dimension()
        client_kwargs = {"model": resolved_model, "api_key": api_key}
        if resolved_model.startswith("text-embedding-3"):
            client_kwargs["dimensions"] = self._expected_dimension
        try:
            self._client = OpenAIEmbeddings(**client_kwargs)
        except TypeError as exc:
            if "dimensions" in client_kwargs:
                raise RuntimeError(
                    "Configured embedding dimension enforcement is not supported by the installed langchain_openai version"
                ) from exc
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._client.embed_documents(list(texts))
        return _validate_embedding_dimensions(
            embeddings,
            expected_dimension=self._expected_dimension,
            label="document_embedding",
        )

    def embed_query(self, text: str) -> List[float]:
        embedding = self._client.embed_query(text)
        return _validate_embedding_dimensions(
            [embedding],
            expected_dimension=self._expected_dimension,
            label="query_embedding",
        )[0]


class GeminiEmbeddingProvider:
    """Gemini-backed embedding provider using the Google GenAI SDK."""

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        resolved_model = (
            model or os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "gemini-embedding-001")
        ).strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini vector embeddings")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for Gemini vector embeddings"
            ) from exc

        self._expected_dimension = _expected_dimension()
        self._batch_size = _batch_size()
        self._model = resolved_model
        self._client = genai.Client(api_key=api_key)
        self._config = types.EmbedContentConfig(
            output_dimensionality=self._expected_dimension,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings: List[List[float]] = []
        items = list(texts)
        for start in range(0, len(items), self._batch_size):
            batch = items[start : start + self._batch_size]
            result = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=self._config,
            )
            embeddings.extend(list(item.values) for item in (result.embeddings or []))
        return _validate_embedding_dimensions(
            embeddings,
            expected_dimension=self._expected_dimension,
            label="document_embedding",
        )

    def embed_query(self, text: str) -> List[float]:
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=self._config,
        )
        embeddings = [list(item.values) for item in (result.embeddings or [])]
        if not embeddings:
            raise RuntimeError("Gemini embedding response did not contain any embeddings")
        embedding = embeddings[0]
        return _validate_embedding_dimensions(
            [embedding],
            expected_dimension=self._expected_dimension,
            label="query_embedding",
        )[0]


def get_embedding_provider(provider: str | None = None) -> EmbeddingProvider:
    """Return the default embedding provider for vector knowledge."""

    resolved_provider = _resolved_provider(provider)
    if resolved_provider == "gemini":
        return GeminiEmbeddingProvider()
    return OpenAIEmbeddingProvider()
