"""Final response-length policy applied after grounded answer generation."""

from __future__ import annotations

import re

from models import AnswerMode, QueryContext

BRIEF_TARGET_WORDS = 90
BRIEF_MAX_WORDS = 140
_MIN_SENTENCE_BOUNDARY_WORDS = 55
_WORD_RE = re.compile(r"\S+")
_SENTENCE_END_RE = re.compile(r"[.!?](?:[\"')\]]*)(?=\s|$)")
_BRIEF_EXEMPT_SOURCES = frozenset(
    {
        "absence_claim_guardrail",
        "citation_gate_fallback",
        "clarification_request",
        "evidence_unavailable",
        "structured_summary_grounding_fallback",
    }
)


def count_words(text: str) -> int:
    """Count display words using the same whitespace rule as the hard cap."""
    return len(_WORD_RE.findall(str(text or "")))


def _clip_to_brief_limit(text: str) -> str:
    word_matches = list(_WORD_RE.finditer(text))
    if len(word_matches) <= BRIEF_MAX_WORDS:
        return text

    hard_end = word_matches[BRIEF_MAX_WORDS - 1].end()
    candidate = text[:hard_end]
    selected_end = hard_end
    for boundary in reversed(list(_SENTENCE_END_RE.finditer(candidate))):
        boundary_text = candidate[: boundary.end()]
        last_token = _WORD_RE.findall(boundary_text)[-1]
        if re.fullmatch(r"\d+\.", last_token):
            continue
        if count_words(boundary_text) >= _MIN_SENTENCE_BOUNDARY_WORDS:
            selected_end = boundary.end()
            break

    clipped = text[:selected_end].rstrip(" \t\r\n,;:-")
    # Avoid leaving common paired Markdown markers open when clipping a long
    # deterministic or model-generated response.
    if clipped.count("**") % 2:
        clipped += "**"
    if clipped.count("`") % 2:
        clipped += "`"
    return clipped + "…"


def _requires_complete_enumeration(ctx: QueryContext) -> bool:
    analysis = ctx.question_analysis
    if analysis is None:
        return False
    classification = getattr(analysis, "classification", None)
    query_type = getattr(getattr(classification, "query_type", None), "value", "")
    if query_type == "regulatory_procedure":
        return True
    answer_kind = getattr(getattr(analysis, "answer_kind", None), "value", "")
    return ctx.response_mode == "knowledge_primary" and answer_kind == "list"


def apply_answer_mode_policy(ctx: QueryContext) -> bool:
    """Apply Brief mode without changing evidence, charts, or terminal guidance.

    Returns True only when the visible summary was shortened.
    """
    if ctx.answer_mode != AnswerMode.BRIEF.value:
        return False
    if ctx.summary_source in _BRIEF_EXEMPT_SOURCES:
        return False
    if _requires_complete_enumeration(ctx):
        return False
    shortened = _clip_to_brief_limit(str(ctx.summary or ""))
    if shortened == ctx.summary:
        return False
    ctx.summary = shortened
    return True
