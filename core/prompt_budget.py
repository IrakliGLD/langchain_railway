"""
Prompt budget enforcement and section-aware truncation.

Extracted from ``core/llm.py`` (Q3a, 2026-06-10) as a pure structural move.

Patch-surface note: tests monkeypatch ``core.llm._enforce_prompt_budget`` --
that works because every production caller is an entry point that REMAINS in
``core.llm`` and resolves the name through that module's globals. Budgets are
passed in as arguments by the entry points; nothing here reads the
ANALYZER/SUMMARIZER/FAST_MODE budget constants that tests patch on core.llm.
"""
import logging
import re
from typing import Optional

from config import PROMPT_BUDGET_MAX_CHARS
from contracts.question_analysis import AnswerKind, QuestionAnalysis

log = logging.getLogger("Enai")


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - 24)
    return text[:keep] + "\n...[truncated]"


# Sections truncated first → last when prompt exceeds budget.
# Sections NOT listed here (user_question) are fully protected.
# Data-primary: sacrifice knowledge before data.
_TRUNCATION_PRIORITY_DATA = [
    "UNTRUSTED_CONVERSATION_HISTORY",
    "UNTRUSTED_DOMAIN_KNOWLEDGE",
    "UNTRUSTED_EXTERNAL_SOURCE_PASSAGES",
    "UNTRUSTED_DATA_PREVIEW",
    "UNTRUSTED_STATISTICS",
]
# Knowledge-primary: sacrifice data before knowledge.
_TRUNCATION_PRIORITY_KNOWLEDGE = [
    "UNTRUSTED_CONVERSATION_HISTORY",
    "UNTRUSTED_DATA_PREVIEW",
    "UNTRUSTED_STATISTICS",
    "UNTRUSTED_DOMAIN_KNOWLEDGE",
    "UNTRUSTED_EXTERNAL_SOURCE_PASSAGES",
]
# Phase D: explanation answers rely on strict grounding against derived metrics
# (correlations, mom, yoy). Shed knowledge before shedding the statistics,
# to ensure the grounding checks don't fail when the budget is squeezed.
_TRUNCATION_PRIORITY_EXPLANATION = [
    "UNTRUSTED_CONVERSATION_HISTORY",
    "UNTRUSTED_DOMAIN_KNOWLEDGE",
    "UNTRUSTED_EXTERNAL_SOURCE_PASSAGES",
    "UNTRUSTED_DATA_PREVIEW",
    "UNTRUSTED_STATISTICS",
]
# Phase D: forecast / scenario answers MUST keep the stats + data preview
# (the projection or payoff is computed against those rows).  Sacrifice
# knowledge passages first — they only add caveats.
_TRUNCATION_PRIORITY_FORECAST_SCENARIO = [
    "UNTRUSTED_CONVERSATION_HISTORY",
    "UNTRUSTED_EXTERNAL_SOURCE_PASSAGES",
    "UNTRUSTED_DOMAIN_KNOWLEDGE",
    "UNTRUSTED_DATA_PREVIEW",
    "UNTRUSTED_STATISTICS",
]
# Default (backward-compatible) — preserves original ordering for callers
# that don't pass an explicit truncation_priority (e.g. llm_summarize legacy path).
_TRUNCATION_PRIORITY = [
    "UNTRUSTED_CONVERSATION_HISTORY",
    "UNTRUSTED_DATA_PREVIEW",
    "UNTRUSTED_DOMAIN_KNOWLEDGE",
    "UNTRUSTED_EXTERNAL_SOURCE_PASSAGES",
    "UNTRUSTED_STATISTICS",
]


def _select_summarizer_truncation_priority(
    *,
    question_analysis: Optional["QuestionAnalysis"] = None,
    effective_answer_kind: Optional[AnswerKind] = None,
    response_mode: str = "",
    resolution_policy: str = "",
) -> list[str]:
    """Pick the summarizer-truncation profile for a given answer.

    Phase D promotes profile selection from the coarse ``response_mode`` axis
    (data vs. knowledge) to the fine ``answer_kind`` axis when the analyzer
    is authoritative:

    * FORECAST / SCENARIO → preserve stats + data preview the longest
      (the projection / payoff is computed against those rows).
    * EXPLANATION → balance data + knowledge; shed pre-computed stats first.
    * KNOWLEDGE / CLARIFY → preserve passages + domain knowledge.
    * SCALAR / LIST / TIMESERIES / COMPARISON → preserve data preview +
      statistics (classic DATA profile).

    Fallback to ``response_mode`` / ``resolution_policy`` when no analyzer
    output is available, matching pre-Phase-D behavior.
    """
    if question_analysis is not None and question_analysis.answer_kind is not None:
        ak = question_analysis.answer_kind
    elif effective_answer_kind is not None:
        ak = effective_answer_kind
    else:
        ak = None

    if ak is not None:
        if ak in (AnswerKind.FORECAST, AnswerKind.SCENARIO):
            return _TRUNCATION_PRIORITY_FORECAST_SCENARIO
        if ak == AnswerKind.EXPLANATION:
            return _TRUNCATION_PRIORITY_EXPLANATION
        if ak in (AnswerKind.KNOWLEDGE, AnswerKind.CLARIFY):
            return _TRUNCATION_PRIORITY_KNOWLEDGE
        # Data shapes (SCALAR, LIST, TIMESERIES, COMPARISON).
        return _TRUNCATION_PRIORITY_DATA

    # Analyzer-absent fallback (preserves pre-Phase-D behavior).
    if response_mode == "knowledge_primary" or resolution_policy == "clarify":
        return _TRUNCATION_PRIORITY_KNOWLEDGE
    return _TRUNCATION_PRIORITY_DATA

_SECTION_CONTENT_RE = re.compile(
    r"((?:UNTRUSTED|CONTRACT|RULE)_\w+):\n<<<(.*?)>>>", re.DOTALL,
)


def _head_tail_truncate(prompt: str, budget: int, label: str) -> str:
    """Original head+tail fallback."""
    marker = "\n\n...[prompt budget applied]...\n\n"
    tail_budget = min(4000, max(500, budget // 3))
    head_budget = max(0, budget - tail_budget - len(marker))
    trimmed = prompt[:head_budget] + marker + prompt[-tail_budget:]
    log.warning(
        "Prompt budget applied (head+tail): label=%s original_chars=%s budget=%s",
        label, len(prompt), budget,
    )
    return trimmed


def _protected_section_fallback_truncate(
    prompt: str,
    budget: int,
    label: str,
    priority: list[str],
) -> str:
    """Emergency tagged-section fallback that preserves protected sections intact.

    When section-aware truncation fails, do not slice through tagged prompt
    sections. Instead, drop whole eligible sections in priority order and keep
    the remaining tagged sections plus trailing schema/instructions unchanged.
    If the protected remainder still exceeds budget, return it intact and log
    the overage rather than truncating the contract mid-block.
    """

    matches = list(_SECTION_CONTENT_RE.finditer(prompt))
    if not matches:
        return _head_tail_truncate(prompt, budget, label)

    prefix = prompt[:matches[0].start()]
    suffix = prompt[matches[-1].end():]
    sections = [
        {
            "name": match.group(1),
            "content": match.group(2),
            "keep": True,
        }
        for match in matches
    ]
    eligible = set(priority or _TRUNCATION_PRIORITY)

    def _render() -> str:
        parts: list[str] = []
        if prefix.strip():
            parts.append(prefix.rstrip())
        for section in sections:
            if not section["keep"]:
                continue
            parts.append(f"{section['name']}:\n<<<{section['content']}>>>")
        if suffix.strip():
            parts.append(suffix.strip())
        return "\n\n".join(parts)

    rendered = _render()
    if len(rendered) <= budget:
        return rendered

    dropped_sections: list[str] = []
    for section_name in priority or _TRUNCATION_PRIORITY:
        if len(rendered) <= budget:
            break
        for section in sections:
            if (
                section["keep"]
                and section["name"] == section_name
                and section["name"] in eligible
            ):
                section["keep"] = False
                dropped_sections.append(section_name)
                rendered = _render()
                break

    if len(rendered) > budget:
        log.warning(
            "Prompt budget fallback preserved protected sections over budget: label=%s final=%d budget=%d dropped_sections=%s",
            label,
            len(rendered),
            budget,
            dropped_sections,
        )
        return rendered

    log.warning(
        "Prompt budget applied (protected-section fallback): label=%s original=%d final=%d budget=%d dropped_sections=%s",
        label,
        len(prompt),
        len(rendered),
        budget,
        dropped_sections,
    )
    return rendered


def _section_aware_truncate(
    prompt: str,
    budget: int,
    label: str,
    priority: list[str] | None = None,
) -> str:
    """Parse prompt into tagged sections and truncate low-priority ones first."""
    # Collect section name, content-start pos, content-end pos, original content
    section_spans: list[tuple[str, int, int, str]] = []
    replaced: dict[str, str] = {}  # section_name → new content
    for m in _SECTION_CONTENT_RE.finditer(prompt):
        section_spans.append((m.group(1), m.start(2), m.end(2), m.group(2)))

    if not section_spans:
        raise ValueError("No tagged sections found in prompt")

    excess = len(prompt) - budget

    for section_name in (priority or _TRUNCATION_PRIORITY):
        if excess <= 0:
            break
        # Find this section's content
        content = None
        for name, _s, _e, orig in section_spans:
            if name == section_name:
                content = orig
                break
        if content is None or not content.strip():
            continue

        current_len = len(content)
        target_len = max(0, current_len - excess)
        if target_len == 0:
            replaced[section_name] = ""
            excess -= current_len
        else:
            new_content = _truncate_text(content, target_len)
            replaced[section_name] = new_content
            excess -= (current_len - len(new_content))

    if excess > 0:
        raise ValueError(f"Still {excess} chars over budget after truncating all eligible sections")

    # Rebuild: replace in reverse position order so earlier offsets stay valid
    result = prompt
    for name, start, end, _orig in reversed(section_spans):
        if name in replaced:
            result = result[:start] + replaced[name] + result[end:]

    log.warning(
        "Prompt budget applied (section-aware): label=%s original=%d final=%d budget=%d truncated_sections=%s",
        label, len(prompt), len(result), budget, list(replaced.keys()),
    )
    return result
