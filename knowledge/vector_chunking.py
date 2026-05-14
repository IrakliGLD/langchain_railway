"""Chunking helpers for vector-backed document ingestion."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Tuple

from contracts.vector_knowledge import ChunkIngestRecord
from knowledge.vector_reference_parser import (
    parse_outgoing_references,
    parse_section_heading,
)

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
        # Phase B.2 fix: yield empty-body sections too. The chunker still
        # produces no chunks for them (``_split_section_text`` returns []),
        # but the heading-stack ancestry state in ``chunk_markdown_text``
        # needs to advance through chapter headings even when the chapter
        # has no body of its own — common in Georgian regulations where
        # ``## თავი I`` immediately precedes ``### მუხლი 1``.
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
    """Split markdown/text into heading-aware chunks for embedding.

    Phase B.2: every chunk carries the structural fields the resolver
    needs — ``article_number``, ``chapter_number``, ``parent_chapter``,
    ``section_kind`` — plus a parsed list of outbound references in
    ``outgoing_refs``. Chunks that don't sit under a recognised heading
    leave those fields empty and behave exactly like pre-Phase-B chunks.
    """

    chunking = config or ChunkingConfig()
    topics = list(topics or [])
    metadata = dict(metadata or {})
    chunks: List[ChunkIngestRecord] = []
    chunk_index = 0

    # Heading-stack ancestry. As we walk the document in reading order,
    # any chapter heading becomes the current "parent chapter" until a
    # later heading overrides it. Chunks emitted while a chapter is
    # current inherit that chapter as ``parent_chapter`` so the
    # resolver can answer "what chapter is article 14 in?" without
    # joining back to a separate table.
    current_chapter: str = ""

    for section_title, body in _iter_sections(text):
        heading = parse_section_heading(section_title)
        # The chunk itself owns its article_number/chapter_number — but
        # the parent_chapter is the chapter that ENCLOSES it, captured
        # from the running context before this section updates it.
        chunk_article_number = heading.article_number
        chunk_chapter_number = heading.chapter_number
        chunk_section_kind = heading.section_kind
        chunk_parent_chapter = current_chapter

        # Update the running chapter context AFTER capturing parent for
        # this section. A chapter heading's own parent is the chapter
        # that came before it (if any), not itself.
        if heading.section_kind == "chapter" and heading.chapter_number:
            current_chapter = heading.chapter_number

        section_chunks = _split_section_text(body, chunking)
        for body_chunk in section_chunks:
            outgoing_refs = parse_outgoing_references(body_chunk)
            chunks.append(
                ChunkIngestRecord(
                    chunk_index=chunk_index,
                    text_content=body_chunk,
                    token_count=estimate_token_count(body_chunk),
                    section_title=section_title,
                    section_path=section_title,
                    article_number=chunk_article_number,
                    chapter_number=chunk_chapter_number,
                    parent_chapter=chunk_parent_chapter,
                    section_kind=chunk_section_kind,
                    outgoing_refs=outgoing_refs,
                    language=language,
                    topics=topics,
                    metadata=metadata,
                )
            )
            chunk_index += 1
    return chunks
