"""Chunking helpers for vector-backed document ingestion."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Tuple

from contracts.vector_knowledge import ChunkIngestRecord

DEFAULT_TARGET_TOKENS = 650
DEFAULT_OVERLAP_TOKENS = 100

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = DEFAULT_TARGET_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS


def estimate_token_count(text: str) -> int:
    """Cheap token estimate sufficient for chunk sizing."""

    words = re.findall(r"\S+", str(text or ""))
    return max(1, int(len(words) * 1.3)) if words else 0


def _iter_sections(text: str) -> Iterable[Tuple[str, str]]:
    text = str(text or "").strip()
    if not text:
        return []

    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: List[Tuple[str, str]] = []
    if matches[0].start() > 0:
        intro = text[:matches[0].start()].strip()
        if intro:
            sections.append(("", intro))

    for idx, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def _split_section_text(section_text: str, config: ChunkingConfig) -> List[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section_text) if part.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0
    overlap_parts: List[str] = []

    for paragraph in paragraphs:
        para_tokens = estimate_token_count(paragraph)
        if current_parts and current_tokens + para_tokens > config.target_tokens:
            chunks.append("\n\n".join(current_parts).strip())
            if config.overlap_tokens > 0:
                running = 0
                overlap_parts = []
                for existing in reversed(current_parts):
                    running += estimate_token_count(existing)
                    overlap_parts.insert(0, existing)
                    if running >= config.overlap_tokens:
                        break
            else:
                overlap_parts = []
            current_parts = list(overlap_parts)
            current_tokens = sum(estimate_token_count(part) for part in current_parts)
        current_parts.append(paragraph)
        current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())
    return [chunk for chunk in chunks if chunk]


def chunk_markdown_text(
    text: str,
    *,
    language: str = "en",
    topics: list[str] | None = None,
    metadata: dict | None = None,
    config: ChunkingConfig | None = None,
) -> List[ChunkIngestRecord]:
    """Split markdown/text into heading-aware chunks for embedding."""

    chunking = config or ChunkingConfig()
    topics = list(topics or [])
    metadata = dict(metadata or {})
    chunks: List[ChunkIngestRecord] = []
    chunk_index = 0

    for section_title, body in _iter_sections(text):
        section_chunks = _split_section_text(body, chunking)
        for body_chunk in section_chunks:
            chunks.append(
                ChunkIngestRecord(
                    chunk_index=chunk_index,
                    text_content=body_chunk,
                    token_count=estimate_token_count(body_chunk),
                    section_title=section_title,
                    section_path=section_title,
                    language=language,
                    topics=topics,
                    metadata=metadata,
                )
            )
            chunk_index += 1
    return chunks
