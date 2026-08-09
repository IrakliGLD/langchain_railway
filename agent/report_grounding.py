"""Single authoritative numeric-grounding engine for report sections."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, DecimalException, InvalidOperation
from typing import Any, Mapping, Sequence

from contracts.report_evidence import ReportEvidenceItem, ReportEvidenceKind
from contracts.report_sections import (
    ReportDerivedClaim,
    ReportDerivedOperand,
    ReportDirectClaim,
    ReportSectionParagraph,
)

_NUMERIC_PATTERN = re.compile(
    r"(?<![\w.])[-+]?"
    r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?%?(?!\w)(?!\.\d)(?!%)"
)
_PERIOD_PATTERN = re.compile(
    r"(?<![\w])(?P<year>\d{4})[-/]"
    r"(?P<segment>\d{1,2}|[Qq][1-4])"
    r"(?:[-/](?P<day>\d{1,2}))?(?![\w])"
)
_RANGE_SEPARATOR_PATTERN = re.compile(r"(?<=\d)\s*[-–—]\s*(?=[\d.])")
_RANGE_TAIL_PATTERN = r"(?:\s*(?:to|[-–—])\s*[-+]?[\d.,]+%?)?"
_CLAIM_NUMBER_PATTERN = (
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_RATIO_COLUMN_NAMES = {
    "balancing_share",
    "generation_share",
    "import_dependency_ratio",
}
_RATIO_UNITS = {
    "fraction",
    "ratio",
    "share",
    "share (0-1)",
}
_PERCENTAGE_POINT_UNITS = {
    "percentage point",
    "percentage points",
    "pp",
}
# Units that name a scale rather than a physical quantity. Nobody writes "12
# count" or "0.62 ratio", so the prose carries the noun and only the value is
# matched; the claim's coordinate is still verified against the cell.
_DIMENSIONLESS_UNITS = frozenset(
    {"count", "index", "rank"} | _RATIO_UNITS
)
_ADDITIVE_UNITS = {
    "gel",
    "gwh",
    "kwh",
    "kw",
    "mwh",
    "mw",
    "thousand mwh",
    "usd",
}


@dataclass(frozen=True, slots=True)
class _NumericFact:
    value: Decimal
    is_percent: bool
    precision: int


@dataclass(frozen=True, slots=True)
class _PeriodFact:
    value: str


@dataclass(frozen=True, slots=True)
class _YearFact:
    value: int


@dataclass(frozen=True, slots=True)
class _ResolvedOperand:
    value: Decimal
    unit: str


_GroundingFact = _NumericFact | _PeriodFact | _YearFact
_MINIMUM_GROUNDED_YEAR = 1900
_MAXIMUM_GROUNDED_YEAR = 2100
_MAXIMUM_UNGROUNDED_VALUES_PER_PARAGRAPH = 12
_MAXIMUM_UNGROUNDED_HINTS = 32
# ReportSectionParagraph.text enforces this floor; a shorter salvage cannot be
# persisted as a paragraph at all.
_MINIMUM_PARAGRAPH_TEXT_CHARS = 20


def _parse_numeric_token(token: str) -> _NumericFact | None:
    raw = str(token).strip()
    is_percent = raw.endswith("%")
    numeric = raw.removesuffix("%").replace(",", "").lstrip("+")
    try:
        value = Decimal(numeric)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    precision = max(0, -value.as_tuple().exponent)
    return _NumericFact(
        value=value,
        is_percent=is_percent,
        precision=precision,
    )


def _normalized_period_fact(match: re.Match[str]) -> _PeriodFact | None:
    year = match.group("year")
    segment = match.group("segment")
    day = match.group("day")
    if segment.lower().startswith("q"):
        if day is not None:
            return None
        return _PeriodFact(f"{year}-{segment.upper()}")
    month_number = int(segment)
    if not 1 <= month_number <= 12:
        return None
    normalized = f"{year}-{month_number:02d}"
    if day is None:
        return _PeriodFact(normalized)
    day_number = int(day)
    if not 1 <= day_number <= 31:
        return None
    return _PeriodFact(f"{normalized}-{day_number:02d}")


def _grounding_facts_from_text(
    text: str,
) -> set[_GroundingFact]:
    facts: set[_GroundingFact] = set()

    def replace_period(match: re.Match[str]) -> str:
        period_fact = _normalized_period_fact(match)
        if period_fact is None:
            return match.group(0)
        facts.add(period_fact)
        facts.add(_YearFact(int(match.group("year"))))
        return " "

    remaining_text = _PERIOD_PATTERN.sub(replace_period, str(text or ""))
    # A hyphen between two digits separates a range; only a hyphen that does not
    # follow a digit is a sign. Periods are already consumed above.
    remaining_text = _RANGE_SEPARATOR_PATTERN.sub(" to ", remaining_text)
    facts.update(
        {
            fact
            for token in _NUMERIC_PATTERN.findall(remaining_text)
            if (fact := _parse_numeric_token(token)) is not None
        }
    )
    return facts


def _grounding_facts_from_value(
    value: Any,
) -> set[_GroundingFact]:
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, (int, float, Decimal)):
        if isinstance(value, float) and not math.isfinite(value):
            return set()
        fact = _parse_numeric_token(str(value))
        return {fact} if fact is not None else set()
    if isinstance(value, str):
        return _grounding_facts_from_text(value)
    return set()


def _normalize_unit(unit: str) -> str:
    normalized = " ".join(str(unit or "").strip().lower().split())
    normalized = re.sub(r"\s+per\s+", "/", normalized)
    return normalized.replace(" / ", "/").replace("/ ", "/").replace(" /", "/")


def _is_ratio_column(item: ReportEvidenceItem, column: str) -> bool:
    """Return whether a column holds a 0-1 ratio needing percent conversion.

    A declared unit is the authority. The name heuristics below only decide
    columns the manifest left undeclared: a ``share_percent`` column carrying
    "%" is already percent-scaled, and re-scaling it by name would turn 62.0
    into 6200% and reject the one claim a writer could correctly make.
    """

    normalized_unit = _normalize_unit(item.unit_by_column.get(column, ""))
    if normalized_unit:
        return normalized_unit in _RATIO_UNITS
    normalized_column = str(column).strip().lower()
    return (
        normalized_column.startswith("share_")
        or normalized_column in _RATIO_COLUMN_NAMES
    )


def _evidence_grounding_facts(
    item: ReportEvidenceItem,
) -> set[_GroundingFact]:
    facts = _grounding_facts_from_text(item.content)
    for row_index in range(len(item.rows)):
        facts.update(_table_row_grounding_facts(item, row_index))
    return facts


def _table_row_grounding_facts(
    item: ReportEvidenceItem,
    row_index: int,
) -> set[_GroundingFact]:
    if item.kind is not ReportEvidenceKind.TABLE or row_index >= len(item.rows):
        return set()
    facts: set[_GroundingFact] = set()
    for column, value in item.rows[row_index].items():
        value_facts = _grounding_facts_from_value(value)
        facts.update(value_facts)
        if not _is_ratio_column(item, column):
            continue
        for fact in value_facts:
            if (
                not isinstance(fact, _NumericFact)
                or fact.is_percent
                or abs(fact.value) > 1
            ):
                continue
            facts.add(
                _NumericFact(
                    value=fact.value * Decimal(100),
                    is_percent=True,
                    precision=max(0, fact.precision - 2),
                )
            )
    return facts


def build_evidence_grounding_index(
    item_by_ref: Mapping[str, ReportEvidenceItem],
    evidence_refs: set[str],
) -> dict[str, frozenset[_GroundingFact]]:
    """Extract each assigned evidence item's facts once for repeated validation."""

    return {
        ref: frozenset(_evidence_grounding_facts(item_by_ref[ref]))
        for ref in evidence_refs
        if ref in item_by_ref
    }


def observed_period_span(item: ReportEvidenceItem) -> dict[str, str] | None:
    """Return the first and last period literal present in one evidence item."""

    periods: set[str] = set()
    for row in item.rows:
        for value in row.values():
            if not isinstance(value, str):
                continue
            for match in _PERIOD_PATTERN.finditer(value):
                period_fact = _normalized_period_fact(match)
                if period_fact is not None:
                    periods.add(period_fact.value)
    if not periods:
        return None
    ordered = sorted(periods)
    return {"first": ordered[0], "last": ordered[-1]}


def _claim_is_year_reference(claim: _NumericFact) -> bool:
    """Return whether a bare integer reads as a calendar year, not a quantity."""

    return (
        not claim.is_percent
        and claim.precision == 0
        and claim.value == claim.value.to_integral_value()
        and _MINIMUM_GROUNDED_YEAR <= int(claim.value) <= _MAXIMUM_GROUNDED_YEAR
    )


def _is_year_reference(claim: _GroundingFact) -> bool:
    """Return whether a claim is a bare year rather than a dated period."""

    if isinstance(claim, _YearFact):
        return True
    if isinstance(claim, _NumericFact):
        return _claim_is_year_reference(claim)
    return False


def _is_temporal_claim(claim: _GroundingFact) -> bool:
    """Return whether a claim names a period rather than asserting a magnitude."""

    if isinstance(claim, (_PeriodFact, _YearFact)):
        return True
    return _claim_is_year_reference(claim)


def _temporal_evidence_facts(
    facts: set[_GroundingFact] | frozenset[_GroundingFact],
) -> set[_GroundingFact]:
    """Keep only typed period facts.

    A year-like integer is a temporal *claim*, but an evidence cell holding one
    is still a magnitude — capacity of 2000 must never ground a reference to the
    year 2000.
    """

    return {
        fact
        for fact in facts
        if isinstance(fact, (_PeriodFact, _YearFact))
    }


def _grounding_claim_is_supported(
    claim: _GroundingFact,
    evidence_facts: set[_GroundingFact],
) -> bool:
    if isinstance(claim, _YearFact):
        return claim in evidence_facts
    if isinstance(claim, _PeriodFact):
        return claim in evidence_facts
    quantum = Decimal(1).scaleb(-claim.precision)
    for fact in evidence_facts:
        if not isinstance(fact, _NumericFact):
            continue
        if fact.is_percent != claim.is_percent:
            continue
        if fact.value == claim.value:
            return True
        if fact.precision <= claim.precision:
            continue
        try:
            if fact.value.quantize(quantum, rounding=ROUND_HALF_UP) == claim.value:
                return True
        except DecimalException:
            continue
    # Last resort: prose names a period as a bare year ("during 2026") while the
    # evidence carries it only inside a finer period literal ("2026-01").
    return (
        _claim_is_year_reference(claim)
        and _YearFact(int(claim.value)) in evidence_facts
    )


def _resolve_operand(
    operand: ReportDerivedOperand,
    paragraph_refs: set[str],
    item_by_ref: Mapping[str, ReportEvidenceItem],
) -> _ResolvedOperand | None:
    if operand.evidence_ref not in paragraph_refs:
        return None
    item = item_by_ref.get(operand.evidence_ref)
    if item is None or item.kind is not ReportEvidenceKind.TABLE:
        return None
    if operand.row_index >= len(item.rows) or operand.column not in item.columns:
        return None
    raw_value = item.rows[operand.row_index].get(operand.column)
    if isinstance(raw_value, bool) or raw_value is None:
        return None
    try:
        value = Decimal(str(raw_value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    if not value.is_finite():
        return None
    unit = _normalize_unit(item.unit_by_column.get(operand.column, ""))
    if not unit:
        return None
    return _ResolvedOperand(value=value, unit=unit)


def _verified_direct_fact(
    claim: ReportDirectClaim,
    paragraph_refs: set[str],
    item_by_ref: Mapping[str, ReportEvidenceItem],
) -> tuple[_NumericFact, set[_GroundingFact]] | None:
    if claim.evidence_ref not in paragraph_refs:
        return None
    item = item_by_ref.get(claim.evidence_ref)
    if item is None or item.kind is not ReportEvidenceKind.TABLE:
        return None
    if claim.row_index >= len(item.rows) or claim.column not in item.columns:
        return None

    raw_facts = [
        fact
        for fact in _grounding_facts_from_value(
            item.rows[claim.row_index].get(claim.column)
        )
        if isinstance(fact, _NumericFact)
    ]
    displayed = _parse_numeric_token(claim.display_value)
    if len(raw_facts) != 1 or displayed is None:
        return None

    evidence_unit = _normalize_unit(item.unit_by_column.get(claim.column, ""))
    claim_unit = _normalize_unit(claim.unit)
    if not evidence_unit or not claim_unit:
        return None

    raw_fact = raw_facts[0]
    if _is_ratio_column(item, claim.column) and displayed.is_percent:
        expected_fact = _NumericFact(
            value=raw_fact.value * Decimal(100),
            is_percent=True,
            precision=max(0, raw_fact.precision - 2),
        )
        if claim_unit != "%":
            return None
    elif evidence_unit == "%":
        expected_fact = _NumericFact(
            value=raw_fact.value,
            is_percent=True,
            precision=raw_fact.precision,
        )
        if claim_unit != "%":
            return None
    else:
        expected_fact = raw_fact
        if displayed.is_percent or claim_unit != evidence_unit:
            return None

    if not _grounding_claim_is_supported(displayed, {expected_fact}):
        return None
    # A verified cell widens only the temporal identity of its row. Widening
    # sibling magnitudes would let an undeclared number inherit this claim's
    # grounding — including at the wrong unit scale.
    return (
        displayed,
        {
            fact
            for fact in _table_row_grounding_facts(item, claim.row_index)
            if not isinstance(fact, _NumericFact)
        },
    )


def _compute_derived_value(
    claim: ReportDerivedClaim,
    operands: list[_ResolvedOperand],
) -> Decimal | None:
    values = [operand.value for operand in operands]
    units = {operand.unit for operand in operands}
    claim_unit = _normalize_unit(claim.unit)

    if claim.operation in {"sum", "mean", "difference"}:
        if len(units) != 1 or claim_unit not in units:
            return None
        if claim.operation == "sum":
            if claim_unit not in _ADDITIVE_UNITS:
                return None
            return sum(values, start=Decimal(0))
        if claim.operation == "mean":
            return sum(values, start=Decimal(0)) / Decimal(len(values))
        return values[1] - values[0]

    if claim.operation in {"percent_change", "ratio"}:
        if len(units) != 1 or claim_unit != "%":
            return None
        denominator = values[0] if claim.operation == "percent_change" else values[1]
        if denominator == 0:
            return None
        if claim.operation == "percent_change":
            return (values[1] - values[0]) / abs(denominator) * Decimal(100)
        return values[0] / denominator * Decimal(100)

    if claim.operation == "percentage_point_change":
        if claim_unit not in _PERCENTAGE_POINT_UNITS:
            return None
        if not units.issubset(_RATIO_UNITS | {"%"}):
            return None
        percentages = [
            operand.value * Decimal(100)
            if operand.unit in _RATIO_UNITS
            else operand.value
            for operand in operands
        ]
        return percentages[1] - percentages[0]
    return None


def _verified_derived_fact(
    claim: ReportDerivedClaim,
    paragraph_refs: set[str],
    item_by_ref: Mapping[str, ReportEvidenceItem],
) -> tuple[_NumericFact, set[_GroundingFact]] | None:
    operands = [
        _resolve_operand(operand, paragraph_refs, item_by_ref)
        for operand in claim.operands
    ]
    if any(operand is None for operand in operands):
        return None
    displayed = _parse_numeric_token(claim.display_value)
    if displayed is None:
        return None
    try:
        computed = _compute_derived_value(
            claim,
            [operand for operand in operands if operand is not None],
        )
        quantum = Decimal(1).scaleb(-displayed.precision)
        if (
            computed is None
            or computed.quantize(quantum, rounding=ROUND_HALF_UP)
            != displayed.value
        ):
            return None
    except DecimalException:
        return None
    # A verified derivation may name the periods of the rows it spans — that is
    # how a change between two months is normally written. Magnitudes from those
    # rows still need their own coordinate-bound claim.
    operand_period_facts: set[_GroundingFact] = set()
    for operand in claim.operands:
        item = item_by_ref.get(operand.evidence_ref)
        if item is None:
            continue
        operand_period_facts.update(
            _temporal_evidence_facts(
                _table_row_grounding_facts(item, operand.row_index)
            )
        )
    return displayed, operand_period_facts


def _verified_derived_display_value(
    claim: ReportDerivedClaim,
    paragraph_refs: set[str],
    item_by_ref: Mapping[str, ReportEvidenceItem],
) -> str | None:
    operands = [
        _resolve_operand(operand, paragraph_refs, item_by_ref)
        for operand in claim.operands
    ]
    if any(operand is None for operand in operands):
        return None
    declared = _parse_numeric_token(claim.display_value)
    if declared is None:
        return None
    try:
        computed = _compute_derived_value(
            claim,
            [operand for operand in operands if operand is not None],
        )
        if computed is None:
            return None
        quantum = Decimal(1).scaleb(-declared.precision)
        rounded = computed.quantize(quantum, rounding=ROUND_HALF_UP)
    except DecimalException:
        return None
    if rounded == 0:
        rounded = abs(rounded)
    rendered = format(rounded, f".{declared.precision}f")
    return rendered + ("%" if declared.is_percent else "")


def _replace_unique_declared_value(
    text: str,
    claim: ReportDerivedClaim,
    replacement: str,
) -> str | None:
    declared = re.escape(claim.display_value)
    normalized_unit = _normalize_unit(claim.unit)
    if claim.display_value.endswith("%"):
        pattern = rf"(?<![\w.,]){declared}(?!\w)"
    elif normalized_unit in _DIMENSIONLESS_UNITS:
        pattern = rf"(?<![\w.,]){declared}(?!\d)(?![.,]\d)(?!%)(?!\w)"
    else:
        unit_pattern = _claim_unit_pattern(claim.unit)
        pattern = (
            rf"(?<![\w.,]){declared}(?![\d.,])"
            rf"(?=\s+{unit_pattern}(?!\w))"
        )
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if len(matches) != 1:
        return None
    match = matches[0]
    return text[: match.start()] + replacement + text[match.end() :]


def normalize_repairable_derived_claims(
    draft: Any,
    item_by_ref: Mapping[str, ReportEvidenceItem],
) -> tuple[Any, int]:
    """Correct unambiguous derived displays from their verified operands."""
    repaired_count = 0
    repaired_paragraphs: list[ReportSectionParagraph] = []
    for paragraph in draft.paragraphs:
        paragraph_refs = set(paragraph.evidence_refs)
        text = paragraph.text
        claims: list[ReportDerivedClaim] = []
        for claim in paragraph.derived_claims:
            expected = _verified_derived_display_value(
                claim,
                paragraph_refs,
                item_by_ref,
            )
            if expected is None or expected == claim.display_value:
                claims.append(claim)
                continue
            repaired_text = _replace_unique_declared_value(
                text,
                claim,
                expected,
            )
            if repaired_text is None:
                claims.append(claim)
                continue
            text = repaired_text
            claims.append(claim.model_copy(update={"display_value": expected}))
            repaired_count += 1
        repaired_paragraphs.append(
            paragraph.model_copy(
                update={"text": text, "derived_claims": claims}
            )
        )
    if not repaired_count:
        return draft, 0
    return draft.model_copy(update={"paragraphs": repaired_paragraphs}), repaired_count


def drop_unrendered_claims(
    draft: Any,
    item_by_ref: Mapping[str, ReportEvidenceItem],
) -> tuple[Any, int]:
    """Delete verified claims whose value the prose never renders.

    ``DIRECT_CLAIM_NOT_USED`` and ``DERIVED_CLAIM_NOT_USED`` mark a claim that
    is correct about its coordinate but absent from every sentence. The reader
    never sees claim metadata, and the validator adds an unused claim's facts to
    no sentence, so deleting it is provably grounding-neutral — nothing that was
    supported becomes unsupported. Keeping it costs the whole document, and
    REPORT_DOCUMENT_INVALID is not retryable.

    Unverified claims are left alone. Those are writer errors the repair pass
    has to see, not surplus to sweep up. Callers must still re-validate: the
    numeric-finding floor counts claims, so a section can owe numbers after a
    drop.
    """

    dropped = 0
    kept_paragraphs: list[ReportSectionParagraph] = []
    for paragraph in draft.paragraphs:
        paragraph_refs = set(paragraph.evidence_refs)
        sentences = _paragraph_sentences(paragraph.text)
        direct_claims: list[ReportDirectClaim] = []
        for direct_claim in paragraph.direct_claims:
            if (
                _verified_direct_fact(
                    direct_claim,
                    paragraph_refs,
                    item_by_ref,
                )
                is not None
                and not any(
                    _direct_claim_appears(direct_claim, sentence)
                    for sentence in sentences
                )
            ):
                dropped += 1
                continue
            direct_claims.append(direct_claim)
        derived_claims: list[ReportDerivedClaim] = []
        for derived_claim in paragraph.derived_claims:
            if (
                _verified_derived_fact(
                    derived_claim,
                    paragraph_refs,
                    item_by_ref,
                )
                is not None
                and not any(
                    _derived_claim_appears(derived_claim, sentence)
                    for sentence in sentences
                )
            ):
                dropped += 1
                continue
            derived_claims.append(derived_claim)
        kept_paragraphs.append(
            paragraph.model_copy(
                update={
                    "direct_claims": direct_claims,
                    "derived_claims": derived_claims,
                }
            )
        )
    if not dropped:
        return draft, 0
    return draft.model_copy(update={"paragraphs": kept_paragraphs}), dropped


def build_derived_claim_repair_hints(
    sections: Sequence[Any],
    item_by_ref: Mapping[str, ReportEvidenceItem],
) -> list[dict[str, Any]]:
    """Project code-computed derived values for the report repair prompt."""
    hints: list[dict[str, Any]] = []
    for section in sections:
        for paragraph_index, paragraph in enumerate(section.paragraphs):
            paragraph_refs = set(paragraph.evidence_refs)
            for claim_index, claim in enumerate(paragraph.derived_claims):
                expected = _verified_derived_display_value(
                    claim,
                    paragraph_refs,
                    item_by_ref,
                )
                if expected is None:
                    continue
                hints.append(
                    {
                        "section_id": section.section_id,
                        "paragraph_index": paragraph_index,
                        "claim_index": claim_index,
                        "operation": claim.operation,
                        "operands": [
                            operand.model_dump(mode="json")
                            for operand in claim.operands
                        ],
                        "verified_display_value": expected,
                        "unit": claim.unit,
                    }
                )
    return hints


def _derived_claim_appears(
    claim: ReportDerivedClaim,
    paragraph_text: str,
) -> bool:
    rendered, _unit_present = _rendered_claim_positions(
        claim.display_value,
        claim.unit,
        paragraph_text,
    )
    return rendered


def _numbers_agree(left: _NumericFact, right: _NumericFact) -> bool:
    """Return whether two renderings denote the same value.

    Grounding already accepts a finer evidence value as support for a coarser
    claim. The prose check has to use the same rule: a report writing
    "141.0 GEL/MWh" for a cell holding 140.9935 is rendering the claim it
    declared, not asserting a different number.
    """

    if left.is_percent != right.is_percent:
        return False
    if left.value == right.value:
        return True
    coarse, fine = (
        (left, right) if left.precision <= right.precision else (right, left)
    )
    quantum = Decimal(1).scaleb(-coarse.precision)
    try:
        return fine.value.quantize(quantum, rounding=ROUND_HALF_UP) == coarse.value
    except DecimalException:
        return False


def _claim_unit_pattern(claim_unit: str) -> str:
    normalized_unit = _normalize_unit(claim_unit)
    unit_parts = [re.escape(part) for part in normalized_unit.split("/") if part]
    unit_pattern = r"\s*(?:/|\bper\b)\s*".join(unit_parts)
    return unit_pattern.replace(r"\ ", r"\s+")


def _rendered_claim_positions(
    display_value: str,
    claim_unit: str,
    paragraph_text: str,
) -> tuple[bool, bool]:
    """Return (value rendered beside its unit, unit rendered anywhere)."""

    declared = _parse_numeric_token(display_value)
    if declared is None:
        return False, False

    # Each candidate sits inside a lookahead so matches may overlap. A compact
    # range like "120.0-130.0 GEL/MWh" is one consuming match, which would hide
    # the second endpoint from every later claim.
    if display_value.endswith("%"):
        candidate_pattern = rf"(?<![\w.,])(?=({_CLAIM_NUMBER_PATTERN}%)(?!\w))"
        unit_present = True
    elif _normalize_unit(claim_unit) in _DIMENSIONLESS_UNITS:
        # Nobody writes "12 count". A dimensionless claim carries its noun in
        # the prose, so only the value is matched — the cell is still verified.
        # The tail guard must reject a longer number ("0.6" inside "0.62",
        # "1" inside "1,234") without rejecting the full stop that ends a
        # sentence, which is why it tests for a digit after the separator
        # rather than for the separator alone.
        candidate_pattern = (
            rf"(?<![\w.,])(?=({_CLAIM_NUMBER_PATTERN})"
            rf"(?!\d)(?![.,]\d)(?!%)(?!\w))"
        )
        unit_present = True
    else:
        unit_pattern = _claim_unit_pattern(claim_unit)
        candidate_pattern = (
            rf"(?<![\w.,])(?=({_CLAIM_NUMBER_PATTERN})(?![\d.,])"
            rf"{_RANGE_TAIL_PATTERN}\s+{unit_pattern}(?!\w))"
        )
        unit_present = (
            re.search(rf"{unit_pattern}(?!\w)", paragraph_text, flags=re.IGNORECASE)
            is not None
        )

    for match in re.finditer(candidate_pattern, paragraph_text, flags=re.IGNORECASE):
        rendered = _parse_numeric_token(match.group(1))
        if rendered is not None and _numbers_agree(declared, rendered):
            return True, unit_present
    return False, unit_present


def _direct_claim_appears(
    claim: ReportDirectClaim,
    paragraph_text: str,
) -> bool:
    rendered, _unit_present = _rendered_claim_positions(
        claim.display_value,
        claim.unit,
        paragraph_text,
    )
    return rendered


def _paragraph_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]


def _paragraph_grounding_support(
    paragraph: ReportSectionParagraph,
    item_by_ref: Mapping[str, ReportEvidenceItem],
    evidence_facts_by_ref: Mapping[
        str,
        frozenset[_GroundingFact],
    ]
    | None,
) -> tuple[
    list[str],
    list[set[_GroundingFact]],
    set[_GroundingFact],
    list[str],
]:
    """Resolve which facts each sentence may rely on, plus claim-shape errors.

    Validation and repair-hint building must agree on what counts as supported;
    computing it twice is how the gate and the hint that explains it drift
    apart. Returns the sentences, their supporting facts, the table facts held
    back for temporal fallback, and the typed-claim errors found on the way.
    """

    errors: list[str] = []
    paragraph_refs = set(paragraph.evidence_refs)
    grounding_index = evidence_facts_by_ref
    if grounding_index is None:
        grounding_index = build_evidence_grounding_index(
            item_by_ref,
            paragraph_refs,
        )
    narrative_facts = set().union(
        *(
            grounding_index[ref]
            for ref in paragraph_refs
            if ref in grounding_index
            and item_by_ref[ref].kind is not ReportEvidenceKind.TABLE
        )
    )
    table_facts = set().union(
        *(
            grounding_index[ref]
            for ref in paragraph_refs
            if ref in grounding_index
            and item_by_ref[ref].kind is ReportEvidenceKind.TABLE
        )
    )
    sentences = _paragraph_sentences(paragraph.text)
    sentence_facts = [set(narrative_facts) for _ in sentences]

    for claim in paragraph.direct_claims:
        direct_fact = _verified_direct_fact(
            claim,
            paragraph_refs,
            item_by_ref,
        )
        if direct_fact is None:
            errors.append("DIRECT_CLAIM_INVALID")
            continue
        matching_sentences = [
            index
            for index, sentence in enumerate(sentences)
            if _direct_claim_appears(claim, sentence)
        ]
        if not matching_sentences:
            # Split the failure so the section diagnostic says which half broke:
            # a value the prose never rendered beside its unit, or a unit the
            # prose never rendered at all. Both look identical otherwise, and
            # that ambiguity cost several production iterations.
            errors.append("DIRECT_CLAIM_NOT_USED")
            if not _rendered_claim_positions(
                claim.display_value,
                claim.unit,
                paragraph.text,
            )[1]:
                errors.append("DIRECT_CLAIM_UNIT_NOT_RENDERED")
            continue
        displayed, row_facts = direct_fact
        unit_facts = _grounding_facts_from_text(claim.unit)
        for index in matching_sentences:
            sentence_facts[index].add(displayed)
            sentence_facts[index].update(row_facts)
            # Some verified unit labels carry numeric notation as part of the
            # unit itself (for example ``share (0-1)``). Those literals are
            # metadata from the verified table column, not independent prose
            # claims, so ground them only in the sentence that renders the
            # corresponding direct claim.
            sentence_facts[index].update(unit_facts)

    for claim in paragraph.derived_claims:
        derived_fact = _verified_derived_fact(
            claim,
            paragraph_refs,
            item_by_ref,
        )
        if derived_fact is None:
            errors.append("DERIVED_CLAIM_INVALID")
            continue
        matching_sentences = [
            index
            for index, sentence in enumerate(sentences)
            if _derived_claim_appears(claim, sentence)
        ]
        if not matching_sentences:
            errors.append("DERIVED_CLAIM_NOT_USED")
            continue
        displayed, operand_period_facts = derived_fact
        for index in matching_sentences:
            sentence_facts[index].add(displayed)
            sentence_facts[index].update(operand_period_facts)

    return sentences, sentence_facts, table_facts, errors


def _unsupported_sentence_claims(
    sentence: str,
    supported_facts: set[_GroundingFact],
    table_facts: set[_GroundingFact],
) -> list[_GroundingFact]:
    """Return the facts a sentence asserts that its evidence cannot support."""

    claims = _grounding_facts_from_text(sentence)
    temporal_support = supported_facts | _temporal_evidence_facts(table_facts)
    if claims and all(_is_temporal_claim(claim) for claim in claims):
        supported_facts = temporal_support
    return [
        claim
        for claim in claims
        if not _grounding_claim_is_supported(
            claim,
            # A bare year is a period reference whatever sits beside it. The
            # sentence-level allowance above required *every* claim to be
            # temporal, so "Coverage spans 138 monthly observations through
            # 2026" flagged the year as a magnitude because the 138 shared the
            # sentence — job c3138586 spent a repair call clearing exactly
            # that. A full *period* is deliberately not given the same
            # latitude: "in 2026-01 the price was X" binds X to a row, and
            # relaxing it would stop catching a value cited against the wrong
            # one. Relief still comes only from periods the evidence carries,
            # so a year the data never mentions is flagged as before.
            temporal_support
            if _is_year_reference(claim)
            else supported_facts,
        )
    ]


def _render_grounding_fact(fact: _GroundingFact) -> str:
    if isinstance(fact, _YearFact):
        return str(fact.value)
    if isinstance(fact, _PeriodFact):
        return fact.value
    rendered = format(fact.value, f".{fact.precision}f")
    return rendered + ("%" if fact.is_percent else "")


def build_ungrounded_claim_repair_hints(
    sections: Sequence[Any],
    item_by_ref: Mapping[str, ReportEvidenceItem],
) -> list[dict[str, Any]]:
    """Name the exact values a repair pass must ground, cite, or drop.

    UNGROUNDED_NUMERIC_CLAIM on its own tells a repairer only that one of its
    numbers is unsupported, leaving it to guess which — and a guess costs the
    whole document, because the code is not retryable. Naming the values is the
    same contract that made derived claims repairable.
    """

    hints: list[dict[str, Any]] = []
    for section in sections:
        for paragraph_index, paragraph in enumerate(section.paragraphs):
            (
                sentences,
                sentence_facts,
                table_facts,
                _,
            ) = _paragraph_grounding_support(paragraph, item_by_ref, None)
            ungrounded: list[str] = []
            for sentence, supported_facts in zip(
                sentences,
                sentence_facts,
                strict=True,
            ):
                ungrounded.extend(
                    _render_grounding_fact(claim)
                    for claim in _unsupported_sentence_claims(
                        sentence,
                        supported_facts,
                        table_facts,
                    )
                )
            if not ungrounded:
                continue
            # The repair prompt raises when it exceeds its char budget, so an
            # unbounded hint list would turn a repairable document into a hard
            # failure. A paragraph with more offenders than this needs rewriting
            # rather than a value-by-value correction.
            hints.append(
                {
                    "section_id": section.section_id,
                    "paragraph_index": paragraph_index,
                    "ungrounded_values": list(dict.fromkeys(ungrounded))[
                        :_MAXIMUM_UNGROUNDED_VALUES_PER_PARAGRAPH
                    ],
                }
            )
            if len(hints) >= _MAXIMUM_UNGROUNDED_HINTS:
                return hints
    return hints


# Bounded because the repair prompt raises when it exceeds its char budget: an
# unbounded coordinate list would turn a repairable document into a hard
# failure, the same trade the ungrounded-value hints make.
_MAXIMUM_CLAIMABLE_HINT_SECTIONS = 6
_MAXIMUM_CLAIMABLE_COORDINATES_PER_SECTION = 12


def _claimable_direct_claim(
    item: ReportEvidenceItem,
    evidence_ref: str,
    row_index: int,
    column: str,
) -> ReportDirectClaim | None:
    """Render one cell the way a valid direct claim must render it.

    A ratio column has to be claimed as a percentage and a percent column keeps
    its sign, so a hint that simply echoed the stored number would advertise a
    claim the validator rejects. The candidate is verified here before it is
    offered, against the same function that will judge the writer's copy of it.
    """

    raw_facts = [
        fact
        for fact in _grounding_facts_from_value(item.rows[row_index].get(column))
        if isinstance(fact, _NumericFact)
    ]
    evidence_unit = _normalize_unit(item.unit_by_column.get(column, ""))
    if len(raw_facts) != 1 or not evidence_unit:
        return None
    fact = raw_facts[0]
    if _is_ratio_column(item, column):
        precision = max(0, fact.precision - 2)
        display_value = f"{fact.value * Decimal(100):.{precision}f}%"
        unit = "%"
    elif evidence_unit == "%":
        display_value = f"{fact.value:.{fact.precision}f}%"
        unit = "%"
    else:
        display_value = f"{fact.value:.{fact.precision}f}"
        unit = item.unit_by_column[column]
    try:
        claim = ReportDirectClaim(
            evidence_ref=evidence_ref,
            row_index=row_index,
            column=column,
            display_value=display_value,
            unit=unit,
        )
    except Exception:
        return None
    if _verified_direct_fact(claim, {evidence_ref}, {evidence_ref: item}) is None:
        return None
    return claim


def build_claimable_coordinate_hints(
    section_specs: Sequence[Any],
    item_by_ref: Mapping[str, ReportEvidenceItem],
) -> list[dict[str, Any]]:
    """Name the exact cells a section short of numeric findings may cite.

    NUMERIC_FINDING_MISSING on its own tells a repairer that it owes a number
    but not which numbers exist, so it either omits the claim again or invents
    one and trades this code for UNGROUNDED_NUMERIC_CLAIM. Naming the citable
    coordinates is the same contract that made derived claims repairable.
    """

    hints: list[dict[str, Any]] = []
    for section_spec in list(section_specs)[:_MAXIMUM_CLAIMABLE_HINT_SECTIONS]:
        coordinates: list[dict[str, Any]] = []
        for evidence_ref in dict.fromkeys(section_spec.required_evidence_refs):
            item = item_by_ref.get(evidence_ref)
            if item is None:
                continue
            for row_index, column, _value in item.citable_numeric_coordinates():
                claim = _claimable_direct_claim(
                    item,
                    evidence_ref,
                    row_index,
                    column,
                )
                if claim is None:
                    continue
                coordinates.append(claim.model_dump(mode="json"))
                if len(coordinates) >= _MAXIMUM_CLAIMABLE_COORDINATES_PER_SECTION:
                    break
            if len(coordinates) >= _MAXIMUM_CLAIMABLE_COORDINATES_PER_SECTION:
                break
        hints.append(
            {
                "section_id": section_spec.section_id,
                "claimable_coordinates": coordinates,
            }
        )
    return hints


def _surviving_paragraph_claims(
    paragraph: ReportSectionParagraph,
    text: str,
    paragraph_refs: set[str],
    item_by_ref: Mapping[str, ReportEvidenceItem],
) -> tuple[list[Any], list[Any]]:
    """Keep the claims still verified and still rendered in the kept text.

    A claim left behind by a dropped sentence becomes CLAIM_NOT_USED, trading
    one blocking error for another.
    """

    direct_claims = [
        claim
        for claim in paragraph.direct_claims
        if _verified_direct_fact(claim, paragraph_refs, item_by_ref) is not None
        and _direct_claim_appears(claim, text)
    ]
    derived_claims = [
        claim
        for claim in paragraph.derived_claims
        if _verified_derived_fact(claim, paragraph_refs, item_by_ref) is not None
        and _derived_claim_appears(claim, text)
    ]
    return direct_claims, derived_claims


def select_grounded_paragraphs(
    draft: Any,
    item_by_ref: Mapping[str, ReportEvidenceItem],
) -> tuple[Any, int]:
    """Drop the sentences a paragraph cannot support, plus the claims they carried.

    Repair rather than discard: one unsupported figure should cost its own
    sentence, not the entire report. ``REPORT_DOCUMENT_INVALID`` is not
    retryable, so the alternative to a shorter section is no report at all.
    Mirrors :func:`agent.summary_grounding.select_grounded_claims` for Stage-4
    summaries.

    Fails closed. A paragraph whose surviving text is too short to be a
    paragraph is dropped, and a draft left with no paragraphs is returned
    unchanged so the caller's validation rejects it exactly as before.
    """

    dropped = 0
    kept_paragraphs: list[ReportSectionParagraph] = []
    seen_texts: set[str] = set()
    for paragraph in draft.paragraphs:
        (
            sentences,
            sentence_facts,
            table_facts,
            _,
        ) = _paragraph_grounding_support(paragraph, item_by_ref, None)
        kept_sentences = [
            sentence
            for sentence, supported_facts in zip(
                sentences,
                sentence_facts,
                strict=True,
            )
            if not _unsupported_sentence_claims(
                sentence,
                supported_facts,
                table_facts,
            )
        ]
        if len(kept_sentences) == len(sentences):
            if paragraph.text not in seen_texts:
                seen_texts.add(paragraph.text)
                kept_paragraphs.append(paragraph)
            continue
        dropped += len(sentences) - len(kept_sentences)
        text = " ".join(kept_sentences).strip()
        if len(text) < _MINIMUM_PARAGRAPH_TEXT_CHARS or text in seen_texts:
            continue
        direct_claims, derived_claims = _surviving_paragraph_claims(
            paragraph,
            text,
            set(paragraph.evidence_refs),
            item_by_ref,
        )
        seen_texts.add(text)
        kept_paragraphs.append(
            paragraph.model_copy(
                update={
                    "text": text,
                    "direct_claims": direct_claims,
                    "derived_claims": derived_claims,
                }
            )
        )
    if not dropped or not kept_paragraphs:
        return draft, dropped
    return draft.model_copy(update={"paragraphs": kept_paragraphs}), dropped


def validate_paragraph_grounding(
    paragraph: ReportSectionParagraph,
    item_by_ref: Mapping[str, ReportEvidenceItem],
    *,
    evidence_facts_by_ref: Mapping[
        str,
        frozenset[_GroundingFact],
    ]
    | None = None,
) -> list[str]:
    """Validate direct and explicitly typed derived facts for one paragraph."""

    (
        sentences,
        sentence_facts,
        table_facts,
        errors,
    ) = _paragraph_grounding_support(
        paragraph,
        item_by_ref,
        evidence_facts_by_ref,
    )
    for sentence, supported_facts in zip(
        sentences,
        sentence_facts,
        strict=True,
    ):
        if _unsupported_sentence_claims(
            sentence,
            supported_facts,
            table_facts,
        ):
            errors.append("UNGROUNDED_NUMERIC_CLAIM")
            break
    return list(dict.fromkeys(errors))
