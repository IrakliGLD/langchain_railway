"""Single authoritative numeric-grounding engine for report sections."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, DecimalException, InvalidOperation
from typing import Any, Mapping

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
_DIMENSIONLESS_UNITS = frozenset({"count", "index", "rank"})
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
    normalized_column = str(column).strip().lower()
    normalized_unit = _normalize_unit(item.unit_by_column.get(column, ""))
    return (
        normalized_unit in _RATIO_UNITS
        or normalized_column.startswith("share_")
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


def _derived_claim_appears(
    claim: ReportDerivedClaim,
    paragraph_text: str,
) -> bool:
    display_pattern = re.escape(claim.display_value)
    if claim.display_value.endswith("%"):
        pattern = rf"(?<![\w.,]){display_pattern}(?!\w)"
    elif _normalize_unit(claim.unit) in _DIMENSIONLESS_UNITS:
        # Same rule as a dimensionless direct claim: the noun lives in the
        # prose, so nobody writes "12 count".
        pattern = rf"(?<![\w.,]){display_pattern}(?![\d.,])(?!\w)"
    else:
        unit_pattern = re.escape(claim.unit).replace(r"\ ", r"\s+")
        pattern = (
            rf"(?<![\w.,]){display_pattern}(?![\d.,]){_RANGE_TAIL_PATTERN}"
            rf"\s+{unit_pattern}(?!\w)"
        )
    return re.search(pattern, paragraph_text, flags=re.IGNORECASE) is not None


def _direct_claim_appears(
    claim: ReportDirectClaim,
    paragraph_text: str,
) -> bool:
    display_pattern = re.escape(claim.display_value)
    if claim.display_value.endswith("%"):
        pattern = rf"(?<![\w.,]){display_pattern}(?!\w)"
    elif _normalize_unit(claim.unit) in _DIMENSIONLESS_UNITS:
        # Nobody writes "12 count". A dimensionless claim carries its noun in
        # the prose, so only the value is matched — the cell is still verified.
        pattern = rf"(?<![\w.,]){display_pattern}(?![\d.,])(?!\w)"
    else:
        normalized_unit = _normalize_unit(claim.unit)
        unit_parts = [
            re.escape(part)
            for part in normalized_unit.split("/")
            if part
        ]
        unit_pattern = r"\s*(?:/|\bper\b)\s*".join(unit_parts)
        unit_pattern = unit_pattern.replace(r"\ ", r"\s+")
        pattern = (
            rf"(?<![\w.,]){display_pattern}(?![\d.,]){_RANGE_TAIL_PATTERN}"
            rf"\s+{unit_pattern}(?!\w)"
        )
    return re.search(pattern, paragraph_text, flags=re.IGNORECASE) is not None


def _paragraph_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]


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
            errors.append("DIRECT_CLAIM_NOT_USED")
            continue
        displayed, row_facts = direct_fact
        for index in matching_sentences:
            sentence_facts[index].add(displayed)
            sentence_facts[index].update(row_facts)

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

    for sentence, supported_facts in zip(
        sentences,
        sentence_facts,
        strict=True,
    ):
        claims = _grounding_facts_from_text(sentence)
        if claims and all(_is_temporal_claim(claim) for claim in claims):
            supported_facts.update(_temporal_evidence_facts(table_facts))
        if any(
            not _grounding_claim_is_supported(claim, supported_facts)
            for claim in claims
        ):
            errors.append("UNGROUNDED_NUMERIC_CLAIM")
            break
    return list(dict.fromkeys(errors))
