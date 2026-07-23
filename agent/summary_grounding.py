"""Numeric grounding, claim provenance, and conservative summary gates."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Set

from agent.fixture_candidates import log_fixture_candidate
from config import PROVENANCE_MIN_COVERAGE
from core.llm import SummaryEnvelope
from models import GroundingPolicy, QueryContext
from utils.language import get_grounding_fallback_message
from utils.metrics import metrics
from utils.trace_logging import trace_detail

log = logging.getLogger("Enai")

_NUMBER_PATTERN = re.compile(
    r"(?<![\d.])-?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?%?"
)

_MAX_NORMALIZED_NUMBER_TOKEN_CHARS = 128

_HASHED_NUMBER_TOKEN_PREFIX = "decimal-sha256:"

_PROVENANCE_CONTEXT_COLS = ("date", "month", "year", "entity", "type_tech", "segment")

_MAX_REFS_PER_TOKEN = 4

_MAX_REFS_PER_CLAIM = 12

_MAX_PROVENANCE_CITATIONS = 30

_UNSUPPORTED_ABSENCE_PATTERNS = (
    re.compile(r"\bdid not have\b[^.\n]{0,120}\b(?:recorded|available|present)\b", re.IGNORECASE),
    re.compile(r"\bwere not\b[^.\n]{0,80}\b(?:recorded|available|present)\b", re.IGNORECASE),
    re.compile(r"\bno\b[^.\n]{0,80}\b(?:recorded|available|present)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:recorded|available|present)\b", re.IGNORECASE),
)

_SAFE_LIMITED_AVAILABILITY_PHRASES = (
    "not established from the provided data",
    "not shown in the provided data",
    "not visible in the provided data",
    "the provided data does not establish",
    "the retrieved rows do not establish",
    "this result set does not establish",
)

_DATA_SHAPE_MAPPING_SIGNALS = (
    "data-shape rule",
    "data shape rule",
    "data mapping",
    "is grouped under",
    "are grouped under",
    "does not contain a dedicated",
    "no dedicated",
    "as a proxy for",
    "as the proxy for",
    "proxy column",
    "using as a proxy",
    "in accordance with the data-shape",
)

_COLUMN_CITATION_PATTERN = re.compile(
    r"`(?:price|share|tariff|balancing|contribution|regulated|deregulated)_[a-z_]+`",
    re.IGNORECASE,
)


def _hashed_number_token(canonical_value: str) -> str:
    digest = hashlib.sha256(canonical_value.encode("utf-8")).hexdigest()
    return f"{_HASHED_NUMBER_TOKEN_PREFIX}{digest}"


def _canonicalize_finite_decimal(value: Decimal) -> str:
    """Return an exact, bounded token without fixed-point exponent expansion."""
    sign_bit, raw_digits, raw_exponent = value.as_tuple()
    digits = list(raw_digits)
    exponent = int(raw_exponent)

    if not any(digits):
        return "0"

    # Decimal.normalize() applies the active precision context and can both
    # round long coefficients and expand huge exponents when formatted with
    # ``f``. Strip only exact trailing zeroes so equivalent spellings retain
    # one canonical value without changing its precision.
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1

    digit_text = "".join(str(digit) for digit in digits)
    sign = "-" if sign_bit else ""
    decimal_position = len(digit_text) + exponent

    if exponent >= 0:
        fixed_length = len(sign) + len(digit_text) + exponent
        if fixed_length <= _MAX_NORMALIZED_NUMBER_TOKEN_CHARS:
            return f"{sign}{digit_text}{'0' * exponent}"
    else:
        fixed_length = (
            len(sign) + len(digit_text) + 1
            if decimal_position > 0
            else len(sign) + 2 + (-decimal_position) + len(digit_text)
        )
        if fixed_length <= _MAX_NORMALIZED_NUMBER_TOKEN_CHARS:
            if decimal_position > 0:
                return f"{sign}{digit_text[:decimal_position]}.{digit_text[decimal_position:]}"
            return f"{sign}0.{'0' * (-decimal_position)}{digit_text}"

    adjusted_exponent = exponent + len(digit_text) - 1
    mantissa = digit_text[0]
    if len(digit_text) > 1:
        mantissa = f"{mantissa}.{digit_text[1:]}"
    compact = f"{sign}{mantissa}e{adjusted_exponent:+d}"
    if len(compact) <= _MAX_NORMALIZED_NUMBER_TOKEN_CHARS:
        return compact

    # The source token already contains an oversized coefficient. Hash its
    # exact canonical tuple so it remains matchable/fail-closed without being
    # retained or copied through every provenance index.
    return _hashed_number_token(f"{sign}{digit_text}e{exponent:+d}")


def _normalize_number_token(raw_token: str) -> Optional[str]:
    token = (raw_token or "").strip().replace(",", "")
    if token.endswith("%"):
        token = token[:-1]
    if not token:
        return None
    try:
        numeric = Decimal(token)
    except (InvalidOperation, ValueError):
        normalized = token if len(token) <= _MAX_NORMALIZED_NUMBER_TOKEN_CHARS else _hashed_number_token(token)
    else:
        normalized = _canonicalize_finite_decimal(numeric) if numeric.is_finite() else token
    if not normalized or normalized in {"-", "+", "."}:
        return None
    return normalized


def _extract_number_tokens(text: str) -> Set[str]:
    tokens: Set[str] = set()
    for match in _NUMBER_PATTERN.finditer(text or ""):
        normalized = _normalize_number_token(match.group(0))
        if not normalized:
            continue
        tokens.add(normalized)
    return tokens


def _build_grounding_corpus(ctx: QueryContext) -> str:
    parts = [ctx.preview or "", ctx.stats_hint or ""]
    # Normalize the policy to its .value first: in Python 3.11+ ``str(StrEnum)``
    # returns "GroundingPolicy.EVIDENCE_AWARE", so the old ``str(...) == enum``
    # compare was ALWAYS False for the enum (production) and silently EXCLUDED
    # domain_knowledge/vector from the grounding corpus. EVIDENCE_AWARE answers
    # that cite domain figures (≈55-57 USD/MWh, article refs) then failed
    # grounding (prod trace 663461d6). Tests passed only because they set the
    # policy as the string "evidence_aware", which compared True.
    policy_value = getattr(ctx.grounding_policy, "value", str(ctx.grounding_policy or ""))
    if policy_value == GroundingPolicy.EVIDENCE_AWARE.value:
        parts.append(ctx.summary_domain_knowledge or "")
        parts.append(ctx.vector_knowledge_prompt or "")
    if ctx.df is not None and not ctx.df.empty:
        try:
            parts.append(ctx.df.head(200).to_string(index=False))
        except Exception:  # pragma: no cover - defensive
            pass
    return "\n".join(parts)


def _add_aggregate_tokens(tokens: Set[str], ctx: QueryContext) -> None:
    """Add column-level aggregates (sum/mean/min/max/count) to grounding tokens.

    Any answer that summarizes tabular data legitimately quotes column
    aggregates — a descriptive "what can you say about X" (light mode) cites
    means and totals just as an analyst-mode calculation does. Gating this on
    analyst mode meant those aggregates were absent from the grounding corpus
    for light-mode summaries, so the strict gate rejected correct answers
    (2026-07-09 generation-mix trace: means 820/223/1049 unmatched → citation
    fallback). Aggregates are computed from the real evidence frame, so adding
    them for all modes preserves the gate's anti-hallucination property — a
    fabricated number still cannot match unless it is a genuine column
    aggregate of the data.
    """
    if ctx.df is None or ctx.df.empty:
        return
    for col in ctx.df.select_dtypes(include="number").columns:
        series = ctx.df[col].dropna()
        if series.empty:
            continue
        for val in [series.sum(), series.mean(), series.min(), series.max(), len(series)]:
            tokens.update(_tokenize_cell_value(val))


def _add_evidence_record_tokens(tokens: Set[str], ctx: QueryContext) -> None:
    """Add computed values from structured evidence records to grounding tokens.

    Derived metrics (MoM deltas, scenario results, etc.) produce values that
    may not exist in raw data rows.  By tokenizing the computed fields from
    ``ctx.analysis_evidence``, we let the grounding check recognise these
    values as legitimate — preventing false-positive failures on analytical
    answers.
    """
    if not ctx.analysis_evidence:
        return

    _NUMERIC_KEYS = (
        "current_value",
        "previous_value",
        "absolute_change",
        "percent_change",
        "correlation_value",
        "trend_slope",
        "aggregate_result",
        "baseline_aggregate",
        "delta_aggregate",
        "delta_percent",
        "min_period_value",
        "max_period_value",
        "mean_period_value",
        "positive_sum",
        "negative_sum",
        "positive_count",
        "negative_count",
        "market_component_aggregate",
        "combined_total_aggregate",
        "scenario_factor",
        "scenario_volume",
    )
    for record in ctx.analysis_evidence:
        if not isinstance(record, dict):
            continue
        for key in _NUMERIC_KEYS:
            val = record.get(key)
            if val is not None:
                tokens.update(_tokenize_cell_value(val))
        # Also tokenize source cell values so the grounding check can
        # trace computations back to their inputs.
        for cell in record.get("source_cells", []):
            if not isinstance(cell, dict):
                continue
            for cell_key in ("value", "min_value", "max_value"):
                cell_val = cell.get(cell_key)
                if cell_val is not None:
                    tokens.update(_tokenize_cell_value(cell_val))


def _add_join_provenance_tokens(tokens: Set[str], ctx: QueryContext) -> None:
    """Tokenize numeric metadata from join provenance records.

    ``ctx.join_provenance`` carries row counts and column info for each
    cross-dataset merge.  Row counts may appear in analytical summaries
    (e.g. "merged 24 rows"), so we register them as valid grounding tokens.
    """
    for prov in getattr(ctx, "join_provenance", []):
        if not isinstance(prov, dict):
            continue
        for key in ("primary_rows", "merged_rows"):
            val = prov.get(key)
            if val is not None:
                tokens.update(_tokenize_cell_value(val))


def _build_grounding_tokens(ctx: QueryContext) -> Set[str]:
    tokens = _extract_number_tokens(_build_grounding_corpus(ctx))
    # Expand text-extracted numbers with cell-level tokenization so that
    # ratios in stats_hint (e.g. 0.0666) also register as percentages (6.66).
    expanded: Set[str] = set()
    for t in tokens:
        expanded.update(_tokenize_cell_value(t))
    tokens.update(expanded)
    if ctx.df is not None and not ctx.df.empty:
        for _, series in ctx.df.head(200).items():
            for value in series.tolist():
                tokens.update(_tokenize_cell_value(value))
    elif ctx.rows:
        for row in ctx.rows[:200]:
            for value in row:
                tokens.update(_tokenize_cell_value(value))
    _add_aggregate_tokens(tokens, ctx)
    _add_evidence_record_tokens(tokens, ctx)
    _add_join_provenance_tokens(tokens, ctx)
    _add_rounded_source_variants(tokens)
    return tokens


def _add_rounded_source_variants(tokens: Set[str]) -> None:
    """Emit rounded forms (0/1/2 decimals) of every fractional source token.

    The corpus holds full precision (``1514.2836``) while the LLM quotes a
    rounded figure (``1,514 thousand MWh`` → token ``1514``); exact string
    match then fails even though the answer faithfully rounded a real value
    (2026-07-08 generation-mix trace: grounding rejected a correct answer,
    missing ``1514``, ``1525``, …). Rounding down REAL source numbers lets
    legitimate rounding match while preserving the gate's anti-hallucination
    property — a fabricated number still cannot match unless a genuine source
    value rounds to it. Mutates ``tokens`` in place.
    """
    variants: Set[str] = set()
    for tok in tokens:
        if "." not in tok:
            continue  # integers carry no extra precision to round away
        try:
            dec = Decimal(tok)
        except (InvalidOperation, ValueError):
            continue
        for places in (0, 1, 2):
            # Emit BOTH rounding conventions: Decimal.quantize defaults to
            # banker's rounding (180.65 → 180.6) while LLMs and humans round
            # half-up (180.65 → 180.7); only the half-even variant existed and
            # correct answers failed the strict gate (2026-07-10 report,
            # trace 1d2d2ca6). Both are faithful roundings of a real source
            # value, so the anti-hallucination property is preserved.
            for rounding in (None, ROUND_HALF_UP):
                try:
                    if rounding is None:
                        rounded = dec.quantize(Decimal(1).scaleb(-places))
                    else:
                        rounded = dec.quantize(Decimal(1).scaleb(-places), rounding=rounding)
                except (InvalidOperation, ValueError):
                    continue
                normalized = _normalize_number_token(str(rounded))
                if normalized and len(normalized) > 1:
                    variants.add(normalized)
    tokens.update(variants)


_RATIO_COLUMN_RE = re.compile(r"(?:^share_|_share$|(?:^|_)ratio(?:$|_)|percent|_pct$)", re.IGNORECASE)


def _is_ratio_column(name: Any) -> bool:
    """True when a column name denotes a ratio/share/percent metric."""
    return bool(name) and bool(_RATIO_COLUMN_RE.search(str(name)))


def _build_grounding_tokens_candidate(ctx: QueryContext) -> Set[str]:
    """Candidate grounding corpus for the shadow harness (NOT wired to gating).

    Identical to :func:`_build_grounding_tokens` except raw result-set cells only
    emit x100 percent tokens when their column is a genuine ratio/share column.
    The stats_hint text expansion and the derived aggregate/evidence/join adders
    keep x100 (those values are ratios/percentages by construction). This narrows
    the false-PASS surface where a raw [0,1] value — e.g. a correlation
    coefficient of 0.85 — currently mints a bare percent token like "85" that an
    ungrounded answer can match. Aggregate/evidence tightening is a deliberate
    follow-up knob, kept permissive here so the first comparison isolates the
    raw-cell effect.
    """
    tokens = _extract_number_tokens(_build_grounding_corpus(ctx))
    expanded: Set[str] = set()
    for t in tokens:
        expanded.update(_tokenize_cell_value(t))  # stats_hint ratios keep x100
    tokens.update(expanded)
    if ctx.df is not None and not ctx.df.empty:
        for col_name, series in ctx.df.head(200).items():
            emit = _is_ratio_column(col_name)
            for value in series.tolist():
                tokens.update(_tokenize_cell_value(value, emit_ratio_percent=emit))
    elif ctx.rows:
        cols = list(getattr(ctx, "cols", None) or [])
        for row in ctx.rows[:200]:
            for idx, value in enumerate(row):
                col_name = cols[idx] if idx < len(cols) else ""
                tokens.update(_tokenize_cell_value(value, emit_ratio_percent=_is_ratio_column(col_name)))
    _add_aggregate_tokens(tokens, ctx)
    _add_evidence_record_tokens(tokens, ctx)
    _add_join_provenance_tokens(tokens, ctx)
    return tokens


@dataclass(frozen=True)
class GroundingComparison:
    """Current-vs-candidate grounding decision for one answer (shadow harness)."""

    current_passed: bool
    candidate_passed: bool
    current_ratio: float
    candidate_ratio: float
    threshold: float
    answer_token_count: int
    # Tokens matched under the CURRENT corpus but not the candidate — i.e. the
    # percent tokens the tightening removes (the false-PASS surface).
    divergent_tokens: List[str]

    @property
    def disagree(self) -> bool:
        return self.current_passed != self.candidate_passed


def compare_grounding_policies(envelope: SummaryEnvelope, ctx: QueryContext) -> GroundingComparison:
    """Shadow-compare the current vs candidate grounding decision. Pure; no gating.

    Mirrors :func:`_is_summary_grounded`'s token extraction and threshold, but
    scores the answer against BOTH corpora so callers can log or review where the
    two policies diverge before any cutover.
    """
    raw_policy = ctx.grounding_policy or GroundingPolicy.STRICT_NUMERIC
    grounding_policy = getattr(raw_policy, "value", str(raw_policy))
    claim_text = "\n".join(envelope.claims or [])
    answer_tokens = _extract_number_tokens((envelope.answer or "") + "\n" + claim_text)
    if grounding_policy == GroundingPolicy.NOT_APPLICABLE.value or not answer_tokens:
        return GroundingComparison(True, True, 1.0, 1.0, 0.0, len(answer_tokens), [])

    threshold = 0.7 if grounding_policy == GroundingPolicy.EVIDENCE_AWARE.value else 0.9
    current_src = _build_grounding_tokens(ctx)
    candidate_src = _build_grounding_tokens_candidate(ctx)
    current_matched = {t for t in answer_tokens if t in current_src}
    candidate_matched = {t for t in answer_tokens if t in candidate_src}
    n = max(1, len(answer_tokens))
    current_passed = bool(current_src) and len(current_matched) / n >= threshold
    candidate_passed = bool(candidate_src) and len(candidate_matched) / n >= threshold
    return GroundingComparison(
        current_passed=current_passed,
        candidate_passed=candidate_passed,
        current_ratio=len(current_matched) / n,
        candidate_ratio=len(candidate_matched) / n,
        threshold=threshold,
        answer_token_count=len(answer_tokens),
        divergent_tokens=sorted(current_matched - candidate_matched),
    )


def _has_unsupported_absence_claims(summary: str) -> bool:
    text = str(summary or "").strip()
    if not text:
        return False

    text_lower = text.lower()
    if any(phrase in text_lower for phrase in _SAFE_LIMITED_AVAILABILITY_PHRASES):
        return False
    # Fix F (2026-05-17) — skip the guardrail when the LLM is doing
    # transparent DATA-SHAPE equivalence-mapping (Q2 trace 5a00ee06).
    # See _DATA_SHAPE_MAPPING_SIGNALS comment block above.
    if any(signal in text_lower for signal in _DATA_SHAPE_MAPPING_SIGNALS):
        return False
    if _COLUMN_CITATION_PATTERN.search(text):
        # The summary cites at least one data column by name in
        # backticks — strong signal that the LLM is being transparent
        # about which columns it used and which it could not map.
        # Genuine hallucinated-absence answers don't cite column names.
        return False

    return any(pattern.search(text) for pattern in _UNSUPPORTED_ABSENCE_PATTERNS)


def _apply_absence_claim_guardrail(ctx: QueryContext) -> None:
    """Replace unsupported absence claims with a conservative fallback.

    Blank or omitted cells in a non-empty result set do not prove that an
    entity-period truly lacked a value. This guardrail only applies to LLM
    narrative outputs; deterministic renderers already preserve the evidence
    shape directly.
    """
    if ctx.summary_source not in {"structured_summary", "legacy_text_fallback"}:
        return
    if not ((ctx.df is not None and not ctx.df.empty) or ctx.rows):
        return
    if not _has_unsupported_absence_claims(ctx.summary):
        return

    log.warning("Summary contained unsupported absence claims; replacing with conservative fallback.")
    # Localized so a non-English question doesn't get an English non-answer.
    ctx.summary = get_grounding_fallback_message(getattr(ctx, "lang_code", "") or "en")
    ctx.summary_source = "absence_claim_guardrail"
    ctx.summary_claims = []
    ctx.summary_citations = ["absence_claim_guardrail"]
    ctx.summary_confidence = 0.2


def _serialize_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    return str(value)


def _expand_text_number_token(raw_token: str) -> Set[str]:
    """Expand a number token extracted from free-form text to its rounded
    and unsigned variants.

    Why this exists
    ---------------

    Text sources like ``stats_hint``, ``domain_knowledge``, and
    ``external_source_passages`` are tokenized via plain regex
    (``_extract_number_tokens``).  That returns the literal token —
    so a source value like ``-0.7712345`` produces only the single
    token ``"-0.7712345"``.

    But LLMs naturally round derived values when writing prose:
    they'll cite ``"-0.77"`` or ``"-0.8"`` rather than the full
    ``"-0.7712345"``.  The provenance gate then sees the LLM token
    as "not in source," even though semantically it is.

    DataFrame cells already get rounded-variant expansion via
    ``_tokenize_cell_value``.  This
    helper brings text sources to parity — same expansion logic.

    Expansion rules
    ---------------

    For each raw token, returns the set containing:
      - the original token (always)
      - the value rounded to 0 decimal places
      - the value rounded to 1 decimal place
      - the value rounded to 2 decimal places
      - percentage forms for ratio-like values in [-1, 1]
      - nearest-ten forms for large non-year values
      - unsigned versions of all the above (for "dropped by 5" matching "-5")

    Non-numeric or non-finite tokens pass through unchanged.

    Used by ``_build_claim_provenance`` for stats_hint /
    domain_knowledge / external_source_passages indexing.  Trace
    2026-05-15 58c10c60 confirmed this asymmetry: LLM cited ``-0.77``
    for a trend_slope of ``-0.7712345`` and the provenance gate
    rejected it as ungrounded.
    """
    result: Set[str] = {raw_token}
    try:
        numeric = Decimal(raw_token)
    except (InvalidOperation, ValueError):
        return result
    if not numeric.is_finite():
        return result

    _add_decimal_rounding_variants(result, numeric, include_nearest_ten=True)

    # Text evidence often carries ratios in stats/citation snippets while the
    # model writes the same value as a percent. Keep this aligned with the live
    # grounding-token path, which already expands ratio cells into percentages.
    if abs(numeric) <= 1:
        _add_decimal_rounding_variants(result, numeric * Decimal("100"))

    # Unsigned versions for ALL tokens generated so far.
    unsigned_extra: Set[str] = set()
    for t in result:
        if t.startswith("-"):
            unsigned_extra.add(t[1:])
    result.update(unsigned_extra)
    return result


def _normalized_decimal_token(value: Decimal) -> Optional[str]:
    return _normalize_number_token(str(value))


def _is_year_like_integer(value: Decimal) -> bool:
    if value != value.to_integral_value():
        return False
    absolute_value = abs(value)
    if absolute_value < 1900 or absolute_value > 2100:
        return False
    try:
        int_value = int(absolute_value)
    except (OverflowError, ValueError):
        return False
    return 1900 <= int_value <= 2100


def _add_decimal_rounding_variants(
    result: Set[str],
    numeric: Decimal,
    *,
    include_nearest_ten: bool = False,
) -> None:
    # Both conventions: banker's (quantize default) and half-up (how LLMs and
    # humans round). See _add_rounded_source_variants for the rationale.
    for places in (0, 1, 2):
        for rounding in (None, ROUND_HALF_UP):
            try:
                if rounding is None:
                    rounded = numeric.quantize(Decimal(1).scaleb(-places))
                else:
                    rounded = numeric.quantize(Decimal(1).scaleb(-places), rounding=rounding)
            except (InvalidOperation, ValueError):
                continue
            token = _normalized_decimal_token(rounded)
            if token:
                result.add(token)

    if include_nearest_ten and abs(numeric) >= 100 and not _is_year_like_integer(numeric):
        try:
            rounded_ten = numeric.quantize(Decimal("1E+1"))
        except (InvalidOperation, ValueError):
            return
        token = _normalized_decimal_token(rounded_ten)
        if token:
            result.add(token)


def _rounded_match_variants(token: str) -> List[str]:
    """Return a numeric token rounded to 2 and 1 decimals.

    Used as a fallback in claim matching: the model may quote a source value at a
    different precision than the 1–2-decimal expansion stored in the index (e.g.
    it writes ``44.321`` for source ``44.3214``). Rounding the *claim* token down
    lets it match the indexed ``44.32``. Only the claim side is rounded, so a
    genuinely-absent number (``999``) still finds no match.
    """
    try:
        numeric = Decimal(token)
    except (InvalidOperation, ValueError):
        return []
    if not numeric.is_finite():
        return []
    variants: List[str] = []
    for places in (2, 1):
        # round(Decimal) is banker's rounding; also try half-up so a claim of
        # "44.37" for source "44.365" matches (LLMs round half-up).
        try:
            candidates = [str(round(numeric, places))]
        except (InvalidOperation, ValueError):
            candidates = []
        try:
            candidates.append(str(numeric.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)))
        except (InvalidOperation, ValueError):
            pass
        for raw in candidates:
            candidate = _normalize_number_token(raw)
            if candidate and candidate != token and candidate not in variants:
                variants.append(candidate)
    return variants


def _tokenize_cell_value(value: Any, *, emit_ratio_percent: bool = True) -> Set[str]:
    tokens = _extract_number_tokens(str(value))
    if value is None or isinstance(value, bool):
        return tokens
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return tokens
    if not numeric.is_finite():
        return tokens

    # Support rounding and truncation for all numeric values
    # This ensures that raw data like 37.9913 can be matched by LLM-rounded "37.99" or "38"

    # 1. Direct percentage support for ratio cells (abs <= 1). Gated by
    # emit_ratio_percent so the grounding shadow harness can build a candidate
    # corpus that scopes this x100 expansion to genuine ratio/share columns.
    if emit_ratio_percent and abs(numeric) <= 1:
        percent_raw = numeric * Decimal("100")
        _add_decimal_rounding_variants(tokens, percent_raw)

        # Truncation for ratios
        pr_str = str(percent_raw)
        if "." in pr_str:
            dec_idx = pr_str.find(".")
            for i in [1, 2]:  # Truncate at 1 or 2 decimals
                if len(pr_str) > dec_idx + i:
                    t = _normalize_number_token(pr_str[: dec_idx + i + 1])
                    if t:
                        tokens.add(t)

    # 2. General rounding support for all numbers (including the primary value)
    _add_decimal_rounding_variants(tokens, numeric, include_nearest_ten=True)

    # 3. General truncation
    num_str = str(numeric)
    if "." in num_str:
        dec_idx = num_str.find(".")
        for i in [1, 2]:
            if len(num_str) > dec_idx + i:
                t = _normalize_number_token(num_str[: dec_idx + i + 1])
                if t:
                    tokens.add(t)

    # 4. Unsigned versions for ALL generated tokens (e.g., "dropped by 5" matching "-5")
    unsigned_extra = set()
    for t in tokens:
        if t.startswith("-"):
            unsigned_extra.add(t[1:])
    tokens.update(unsigned_extra)

    return tokens


def _build_row_context(cols: List[Any], row: tuple) -> Dict[str, Any]:
    normalized = [str(c).lower() for c in cols]
    context: Dict[str, Any] = {}
    for key in _PROVENANCE_CONTEXT_COLS:
        if key in normalized:
            idx = normalized.index(key)
            if idx < len(row):
                context[str(cols[idx])] = _serialize_scalar(row[idx])
    if context:
        return context
    for idx, col in enumerate(cols[:3]):
        if idx < len(row):
            context[str(col)] = _serialize_scalar(row[idx])
    return context


def _build_claim_provenance(
    claims: List[str],
    cols: List[Any],
    rows: List[tuple],
    query_hash: str = "",
    source: str = "",
    stats_hint: str = "",
    domain_knowledge: str = "",
    external_source_passages: str = "",
) -> tuple[List[Dict[str, Any]], float, List[str]]:
    token_index: Dict[str, List[Dict[str, Any]]] = {}
    source_priority = {
        "sql": 0,
        "tool": 0,
        "derived_analysis": 1,
        "external_source_passages": 2,
        "domain_knowledge": 3,
        "unknown": 4,
    }

    def _index_text_source(text: str, *, source_name: str, column_label: str, coordinate: str) -> None:
        if not text:
            return
        for raw_token in _extract_number_tokens(text):
            # Build the ref entry once from the RAW token (for human-readable
            # citation display showing the source value as written).
            ref_entry = {
                "source": source_name,
                "row_number": 0,
                "row_index": -1,
                "column": column_label,
                "value": raw_token,
                "cell_id": f"{source_name}:{hashlib.md5(raw_token.encode()).hexdigest()[:8]}",
                "row_context": {"Type": column_label},
                "coordinate": coordinate,
                "query_hash": query_hash,
            }
            # Index under raw_token AND its rounded/unsigned variants so the
            # gate accepts LLM-rounded numbers as grounded.  See the docstring
            # on _expand_text_number_token for the asymmetry this fixes.
            for indexed_token in _expand_text_number_token(raw_token):
                token_index.setdefault(indexed_token, []).append(ref_entry)

    # --- Index statistics and derived analysis ---
    if stats_hint:
        for raw_token in _extract_number_tokens(stats_hint):
            ref_entry = {
                "source": "derived_analysis",
                "row_number": 0,
                "row_index": -1,
                "column": "Statistics/Causal Evidence",
                "value": raw_token,
                "cell_id": f"derived:{hashlib.md5(raw_token.encode()).hexdigest()[:8]}",
                "row_context": {"Type": "System Analysis"},
                "coordinate": "stats_hint",
                "query_hash": query_hash,
            }
            for indexed_token in _expand_text_number_token(raw_token):
                token_index.setdefault(indexed_token, []).append(ref_entry)
    _index_text_source(
        domain_knowledge,
        source_name="domain_knowledge",
        column_label="Domain Knowledge",
        coordinate="domain_knowledge",
    )
    _index_text_source(
        external_source_passages,
        source_name="external_source_passages",
        column_label="External Source Passages",
        coordinate="external_source_passages",
    )
    for row_idx, row in enumerate(rows):
        row_context = _build_row_context(cols, row)
        for col_idx, col_name in enumerate(cols):
            if col_idx >= len(row):
                continue
            value = row[col_idx]
            for token in _tokenize_cell_value(value):
                refs = token_index.setdefault(token, [])
                refs.append(
                    {
                        "row_index": row_idx,
                        "row_number": row_idx + 1,
                        "column": str(col_name),
                        "value": _serialize_scalar(value),
                        "coordinate": f"r{row_idx + 1}.c[{col_name}]",
                        "query_hash": query_hash,
                        "source": source or "unknown",
                        "row_fingerprint": hashlib.sha256(
                            f"{query_hash}|{row_idx + 1}|{repr(row)}".encode("utf-8")
                        ).hexdigest()[:16],
                        "cell_id": f"{query_hash}:r{row_idx + 1}:c[{col_name}]",
                        "row_context": row_context,
                    }
                )

    claim_entries: List[Dict[str, Any]] = []
    citation_anchors: List[str] = []
    numeric_claims = 0
    grounded_numeric_claims = 0

    for claim_idx, claim in enumerate(claims):
        claim_tokens = sorted(_extract_number_tokens(claim))
        if claim_tokens:
            numeric_claims += 1

        matched_tokens: List[str] = []
        unmatched_tokens: List[str] = []
        claim_refs: List[Dict[str, Any]] = []
        seen_refs = set()

        for token in claim_tokens:
            raw_refs = token_index.get(token)
            if not raw_refs:
                # Precision-tolerant retry: the model may round a source value to
                # a different decimal precision than the indexed 1–2-decimal
                # expansion (e.g. 44.321 for source 44.3214).
                for variant in _rounded_match_variants(token):
                    if variant in token_index:
                        raw_refs = token_index[variant]
                        break
            refs = sorted(
                raw_refs or [],
                key=lambda ref: (
                    source_priority.get(str(ref.get("source") or "unknown"), 4),
                    int(ref.get("row_index", -1)),
                ),
            )
            if not refs:
                unmatched_tokens.append(token)
                continue
            matched_tokens.append(token)
            if len(claim_refs) >= _MAX_REFS_PER_CLAIM:
                continue
            for ref in refs[:_MAX_REFS_PER_TOKEN]:
                ref_key = (ref["row_number"], ref["column"], str(ref["value"]))
                if ref_key in seen_refs:
                    continue
                seen_refs.add(ref_key)
                claim_refs.append(ref)
                citation_anchors.append(f"claim_{claim_idx}:{ref['cell_id']}")
                if len(claim_refs) >= _MAX_REFS_PER_CLAIM:
                    break

        if claim_tokens and not unmatched_tokens and claim_refs:
            grounded_numeric_claims += 1

        claim_entries.append(
            {
                "claim_index": claim_idx,
                "claim_text": claim,
                "tokens": claim_tokens,
                "matched_tokens": matched_tokens,
                "unmatched_tokens": unmatched_tokens,
                "cell_refs": claim_refs,
                "is_fully_grounded": bool(claim_tokens) and not unmatched_tokens and bool(claim_refs),
            }
        )

    coverage = 1.0 if numeric_claims == 0 else grounded_numeric_claims / max(1, numeric_claims)
    deduped_anchors = list(dict.fromkeys(citation_anchors))
    return claim_entries, coverage, deduped_anchors


def _derive_claims_from_text(summary_text: str) -> List[str]:
    claims: List[str] = []
    for line in (summary_text or "").splitlines():
        if not line.strip():
            continue
        # List ordinals are presentation, not analytical claims. Remove them
        # before number extraction while retaining every substantive line.
        claim = re.sub(r"^\s*(?:(?:[-*•])\s+|\d+[.)]\s+)", "", line).strip()
        if claim:
            claims.append(claim)
    if not claims and summary_text.strip():
        claims = [summary_text.strip()]
    return claims


def _attach_claim_provenance(ctx: QueryContext) -> None:
    claims = [str(c).strip() for c in (ctx.summary_claims or []) if str(c).strip()]
    source_cols = list(ctx.provenance_cols or ctx.cols or [])
    source_rows = [tuple(r) for r in (ctx.provenance_rows or ctx.rows or [])]

    if not claims:
        ctx.summary_claim_provenance = []
        ctx.summary_provenance_coverage = 0.0
        return
    claim_prov, coverage, anchors = _build_claim_provenance(
        claims,
        source_cols,
        source_rows,
        query_hash=ctx.provenance_query_hash or "unknown",
        source=ctx.provenance_source or "unknown",
        stats_hint=ctx.stats_hint or "",
        domain_knowledge=ctx.summary_domain_knowledge or "",
        external_source_passages=ctx.vector_knowledge_prompt or "",
    )
    ctx.summary_claim_provenance = claim_prov
    ctx.summary_provenance_coverage = round(float(coverage), 4)
    if anchors:
        merged = list(dict.fromkeys(list(ctx.summary_citations or []) + anchors))
        ctx.summary_citations = merged[:_MAX_PROVENANCE_CITATIONS]


def _enforce_provenance_gate(ctx: QueryContext) -> None:
    if ctx.summary_source == "structured_summary_grounding_fallback":
        ctx.summary_provenance_gate_passed = False
        ctx.summary_provenance_gate_reason = "grounding_guardrail_fallback"
        ctx.summary_provenance_coverage = 0.0
        trace_detail(
            log,
            ctx,
            "stage_4_summarize_data",
            "provenance_gate",
            gate_passed=False,
            gate_reason=ctx.summary_provenance_gate_reason,
            numeric_claims=0,
            coverage=0.0,
        )
        return

    claim_entries = list(ctx.summary_claim_provenance or [])
    numeric_claims = [entry for entry in claim_entries if entry.get("tokens")]
    if not claim_entries:
        ctx.summary_provenance_gate_passed = True
        ctx.summary_provenance_gate_reason = "no_claims"
        trace_detail(
            log,
            ctx,
            "stage_4_summarize_data",
            "provenance_gate",
            gate_passed=True,
            gate_reason=ctx.summary_provenance_gate_reason,
            numeric_claims=0,
            coverage=0.0,
        )
        return
    if not numeric_claims:
        ctx.summary_provenance_gate_passed = True
        ctx.summary_provenance_gate_reason = "no_numeric_claims"
        trace_detail(
            log,
            ctx,
            "stage_4_summarize_data",
            "provenance_gate",
            gate_passed=True,
            gate_reason=ctx.summary_provenance_gate_reason,
            numeric_claims=0,
            coverage=float(ctx.summary_provenance_coverage or 0.0),
        )
        return

    has_ungrounded_claim = any(not bool(entry.get("is_fully_grounded")) for entry in numeric_claims)
    coverage = float(ctx.summary_provenance_coverage or 0.0)
    # Use coverage-only gating: the binary has_ungrounded_claim check was
    # too strict for analytical queries where the LLM legitimately derives
    # values (MoM changes, percentages) from raw data.  Coverage already
    # captures grounding quality proportionally.  has_ungrounded_claim is
    # still computed and logged for observability.
    # Align with the grounding check: an EVIDENCE_AWARE (mixed data + domain
    # knowledge) answer is accepted by _is_summary_grounded at 0.70, so the
    # provenance gate must not reject the same answer at 0.80. Normalize the
    # policy value to dodge the Py3.11 StrEnum str() trap.
    _policy_val = getattr(ctx.grounding_policy, "value", str(ctx.grounding_policy or ""))
    min_coverage = (
        min(PROVENANCE_MIN_COVERAGE, 0.70)
        if _policy_val == GroundingPolicy.EVIDENCE_AWARE.value
        else PROVENANCE_MIN_COVERAGE
    )
    # The legacy text fallback has no structured claim contract. Its claims are
    # derived from the complete returned text, so fail closed if even one
    # numeric claim is ungrounded. Structured summaries retain the established
    # coverage policy for explicitly derived analytical values.
    legacy_text_fallback = ctx.summary_source == "legacy_text_fallback"
    gate_passed = coverage >= min_coverage and (not legacy_text_fallback or not has_ungrounded_claim)
    if gate_passed:
        ctx.summary_provenance_gate_passed = True
        ctx.summary_provenance_gate_reason = "ok"
        trace_detail(
            log,
            ctx,
            "stage_4_summarize_data",
            "provenance_gate",
            gate_passed=True,
            gate_reason=ctx.summary_provenance_gate_reason,
            numeric_claims=len(numeric_claims),
            coverage=coverage,
        )
        return

    # When coverage is too low, replace the draft answer with a safe retry prompt.
    metrics.log_summary_grounding_failure()
    if hasattr(metrics, "log_provenance_gate_failure"):
        metrics.log_provenance_gate_failure()
    log_fixture_candidate("provenance_gate_failure", ctx)
    ctx.summary_provenance_gate_passed = False
    ctx.summary_provenance_gate_reason = (
        f"coverage={coverage:.4f}, min={min_coverage:.4f}, ungrounded_numeric_claims={int(has_ungrounded_claim)}"
    )
    ctx.summary = get_grounding_fallback_message(getattr(ctx, "lang_code", "") or "en")
    ctx.summary_source = "citation_gate_fallback"
    ctx.summary_claims = []
    ctx.summary_citations = ["citation_gate_fallback"]
    ctx.summary_confidence = min(float(ctx.summary_confidence or 0.0), 0.2)
    ctx.summary_claim_provenance = []
    ctx.summary_provenance_coverage = 0.0
    unmatched = [
        {
            "claim_index": entry.get("claim_index"),
            "unmatched_tokens": list(entry.get("unmatched_tokens") or []),
        }
        for entry in numeric_claims
        if entry.get("unmatched_tokens")
    ]
    trace_detail(
        log,
        ctx,
        "stage_4_summarize_data",
        "provenance_gate",
        gate_passed=False,
        gate_reason=ctx.summary_provenance_gate_reason,
        numeric_claims=len(numeric_claims),
        coverage=coverage,
        unmatched_tokens=unmatched,
    )
    trace_detail(
        log,
        ctx,
        "stage_4_summarize_data",
        "artifact",
        debug=True,
        summary_claim_provenance=claim_entries,
    )


# Stable public boundary; underscored names remain available for compatibility.
build_grounding_tokens = _build_grounding_tokens
apply_absence_claim_guardrail = _apply_absence_claim_guardrail
derive_claims_from_text = _derive_claims_from_text
attach_claim_provenance = _attach_claim_provenance
enforce_provenance_gate = _enforce_provenance_gate
