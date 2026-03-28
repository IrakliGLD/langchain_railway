"""Runtime loader for LLM skill reference files.

Reads skill reference files at startup, caches content, and provides
helper functions for injecting skill guidance into LLM prompts.

Usage:
    from skills.loader import load_reference, get_answer_template, get_focus_guidance

    # Load a full reference file
    text = load_reference("energy-analyst", "entity-taxonomy.md")

    # Get answer template for a query type
    template = get_answer_template("comparison")

    # Get focus-specific guidance for the summarizer
    guidance = get_focus_guidance("balancing")
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Reference file cache
# ---------------------------------------------------------------------------

_cache: dict[str, str] = {}


def load_reference(skill_name: str, ref_name: str) -> str:
    """Read a skill reference file. Cached after first read.

    Args:
        skill_name: Directory name under skills/ (e.g., "energy-analyst")
        ref_name: Filename under references/ (e.g., "entity-taxonomy.md")

    Returns:
        File content as a string. Empty string if the file does not exist.
    """
    key = f"{skill_name}/{ref_name}"
    if key not in _cache:
        path = SKILLS_DIR / skill_name / "references" / ref_name
        try:
            _cache[key] = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.warning("Skill reference file not found: %s", path)
            _cache[key] = ""
    return _cache[key]


# ---------------------------------------------------------------------------
# Answer template selection
# ---------------------------------------------------------------------------

# Maps query-type strings → section headers in answer-templates.md.
# Keys include both heuristic classify_query_type() values and LLM
# QuestionAnalysis.classification.query_type enum values so the same
# lookup works regardless of the classification source.
_QUERY_TYPE_TO_TEMPLATE_SECTION: dict[str, str] = {
    # Heuristic classify_query_type() values (fallback)
    "single_value": "## Template: factual_lookup / single_value",
    "list": "## Template: data_retrieval / list",
    "comparison": "## Template: comparison",
    "trend": "## Template: data_explanation / driver analysis",
    "table": "## Template: data_retrieval / list",
    "unknown": "## Template: data_explanation / driver analysis",
    # LLM QueryType enum values (primary when QuestionAnalysis available)
    "conceptual_definition": "## Template: conceptual_definition",
    "regulatory_procedure": "## Template: regulatory_procedure",
    "factual_lookup": "## Template: factual_lookup / single_value",
    "data_retrieval": "## Template: data_retrieval / list",
    "data_explanation": "## Template: data_explanation / driver analysis",
    "forecast": "## Template: forecast",
    # "comparison" already mapped above; "ambiguous"/"unsupported" intentionally
    # omitted so they fall through to the full template file.
}


def get_answer_template(query_type: str) -> str:
    """Extract the answer template section for a query type.

    Args:
        query_type: Either a heuristic classify_query_type() value
            (single_value, list, comparison, trend, table, unknown) or an
            LLM QuestionAnalysis.classification.query_type enum value
            (conceptual_definition, factual_lookup, data_retrieval,
            data_explanation, comparison, forecast, ambiguous, unsupported).

    Returns:
        The matching template section text, or a safe default if no match.
    """
    full_text = load_reference("answer-composer", "answer-templates.md")
    if not full_text:
        return ""

    section_header = _QUERY_TYPE_TO_TEMPLATE_SECTION.get(query_type, "")
    if not section_header:
        return "Answer the question clearly and concisely. Use structured formatting where appropriate."

    return _extract_section(full_text, section_header)


# ---------------------------------------------------------------------------
# Focus-specific guidance
# ---------------------------------------------------------------------------

# Maps query-focus strings → section headers in focus-guidance-catalog.md.
# Values come from heuristic get_query_focus() or from vector-chunk
# document_type metadata (e.g. "regulation").
_FOCUS_TO_CATALOG_SECTION: dict[str, str] = {
    "balancing": "## Focus: Balancing",
    "tariff": "## Focus: Tariff",
    "cpi": "## Focus: CPI / Inflation",
    "generation": "## Focus: Generation",
    "trade": "## Focus: Trade",
    "energy_security": "## Focus: Energy Security",
    "regulation": "## Focus: Regulation",
    "general": "",  # no focus-specific section; always-rules still apply
}


def get_focus_guidance(focus: str, skill: str = "answer-composer") -> str:
    """Extract focus-specific guidance from the catalog.

    Args:
        focus: Return value from get_query_focus()
            (balancing, tariff, cpi, generation, trade, energy_security, regulation, general)
        skill: Which skill's catalog to use
            ("answer-composer" for summarizer, "sql-planner" for planner)

    Returns:
        Always-included rules + the focus-specific section.
    """
    catalog_file = (
        "focus-guidance-catalog.md" if skill == "answer-composer"
        else "guidance-catalog.md"
    )
    full_text = load_reference(skill, catalog_file)
    if not full_text:
        return ""

    # Always include unconditional rules
    always_section = _extract_section(full_text, "## Always")

    # Add focus-specific section
    section_header = _FOCUS_TO_CATALOG_SECTION.get(focus, "")
    focus_section = _extract_section(full_text, section_header) if section_header else ""

    parts = [p for p in [always_section, focus_section] if p]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Seasonal-adjusted trend guidance (conditional)
# ---------------------------------------------------------------------------

def get_seasonal_trend_guidance() -> str:
    """Return the seasonal-adjusted trend analysis rules.

    Should be injected when stats_hint contains 'SEASONAL-ADJUSTED TREND ANALYSIS'.
    """
    full_text = load_reference("answer-composer", "focus-guidance-catalog.md")
    if not full_text:
        return ""
    return _extract_section(
        full_text,
        "### Seasonal-adjusted trend analysis",
    )


# ---------------------------------------------------------------------------
# Balancing analysis template (for enriching balancing queries)
# ---------------------------------------------------------------------------

def get_balancing_template() -> str:
    """Return the full balancing analysis template."""
    return load_reference("answer-composer", "balancing-analysis-template.md")


# ---------------------------------------------------------------------------
# Forecast caveats
# ---------------------------------------------------------------------------

def get_forecast_caveats() -> str:
    """Return forecast caveat templates."""
    return load_reference("answer-composer", "forecast-caveats.md")


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

_EXPECTED_FILES: list[tuple[str, str]] = [
    # energy-analyst
    ("energy-analyst", "entity-taxonomy.md"),
    ("energy-analyst", "driver-framework.md"),
    ("energy-analyst", "seasonal-rules.md"),
    ("energy-analyst", "confidentiality-rules.md"),
    ("energy-analyst", "domain-focus-index.md"),
    # sql-planner
    ("sql-planner", "plan-schema.md"),
    ("sql-planner", "chart-strategy-rules.md"),
    ("sql-planner", "guidance-catalog.md"),
    ("sql-planner", "sql-patterns.md"),
    # answer-composer
    ("answer-composer", "answer-templates.md"),
    ("answer-composer", "formatting-rules.md"),
    ("answer-composer", "balancing-analysis-template.md"),
    ("answer-composer", "focus-guidance-catalog.md"),
    ("answer-composer", "forecast-caveats.md"),
]


def validate_skills(*, raise_on_missing: bool = False) -> list[str]:
    """Check that all expected skill reference files exist.

    Args:
        raise_on_missing: If True, raise FileNotFoundError for missing files.

    Returns:
        List of missing file paths (empty if all present).
    """
    missing: list[str] = []
    for skill, ref in _EXPECTED_FILES:
        path = SKILLS_DIR / skill / "references" / ref
        if not path.is_file():
            missing.append(str(path))

    if missing:
        msg = f"Missing {len(missing)} skill reference files: {missing}"
        log.warning(msg)
        if raise_on_missing:
            raise FileNotFoundError(msg)
    else:
        log.info("All %d skill reference files validated.", len(_EXPECTED_FILES))

    return missing


def warmup_cache() -> None:
    """Pre-load all expected reference files into the cache."""
    for skill, ref in _EXPECTED_FILES:
        load_reference(skill, ref)
    log.info("Skill reference cache warmed: %d files loaded.", len(_cache))


# ---------------------------------------------------------------------------
# Content hash for cache-busting
# ---------------------------------------------------------------------------

_content_hash: str = ""


def get_skills_content_hash() -> str:
    """Return a short hash of all cached skill file contents.

    Useful for including in LLM cache keys so that editing a skill
    reference file automatically invalidates stale cached responses.
    The hash is computed once (after warmup or on first call) and
    then memoised for the process lifetime.
    """
    global _content_hash
    if _content_hash:
        return _content_hash

    # Ensure files are loaded
    if not _cache:
        warmup_cache()

    h = hashlib.sha256()
    for key in sorted(_cache):
        h.update(_cache[key].encode("utf-8"))
    _content_hash = h.hexdigest()[:12]
    return _content_hash


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_section(text: str, header: str) -> str:
    """Extract a markdown section from text (header through next same-level header).

    Args:
        text: Full markdown text
        header: The ## or ### header to extract (e.g., "## Focus: Balancing")

    Returns:
        The section content including the header, or empty string if not found.
    """
    if not header or header not in text:
        return ""

    # Determine header level (count leading #)
    level = len(header) - len(header.lstrip("#"))

    start = text.index(header)
    # Find next header at same or higher level
    rest = text[start + len(header):]
    pattern = re.compile(rf"^#{{{1},{level}}}\s", re.MULTILINE)
    match = pattern.search(rest)
    end = start + len(header) + match.start() if match else len(text)

    return text[start:end].strip()
