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
class _ResolvedOperand:
    value: Decimal
    unit: str


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
) -> set[_NumericFact | _PeriodFact]:
    facts: set[_NumericFact | _PeriodFact] = set()

    def replace_period(match: re.Match[str]) -> str:
        period_fact = _normalized_period_fact(match)
        if period_fact is None:
            return match.group(0)
        facts.add(period_fact)
        year_fact = _parse_numeric_token(match.group("year"))
        if year_fact is not None:
            facts.add(year_fact)
        return " "

    remaining_text = _PERIOD_PATTERN.sub(replace_period, str(text or ""))
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
) -> set[_NumericFact | _PeriodFact]:
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
) -> set[_NumericFact | _PeriodFact]:
    facts = _grounding_facts_from_text(item.content)
    for row in item.rows:
        for column, value in row.items():
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
) -> dict[str, frozenset[_NumericFact | _PeriodFact]]:
    """Extract each assigned evidence item's facts once for repeated validation."""

    return {
        ref: frozenset(_evidence_grounding_facts(item_by_ref[ref]))
        for ref in evidence_refs
        if ref in item_by_ref
    }


def _grounding_claim_is_supported(
    claim: _NumericFact | _PeriodFact,
    evidence_facts: set[_NumericFact | _PeriodFact],
) -> bool:
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
    return False


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
) -> _NumericFact | None:
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
    return displayed


def _derived_claim_appears(
    claim: ReportDerivedClaim,
    paragraph_text: str,
) -> bool:
    display_pattern = re.escape(claim.display_value)
    if claim.display_value.endswith("%"):
        pattern = rf"(?<![\w.,]){display_pattern}(?!\w)"
    else:
        unit_pattern = re.escape(claim.unit).replace(r"\ ", r"\s+")
        pattern = (
            rf"(?<![\w.,]){display_pattern}(?![\d.,])"
            rf"\s+{unit_pattern}(?!\w)"
        )
    return re.search(pattern, paragraph_text, flags=re.IGNORECASE) is not None


def validate_paragraph_grounding(
    paragraph: ReportSectionParagraph,
    item_by_ref: Mapping[str, ReportEvidenceItem],
    *,
    evidence_facts_by_ref: Mapping[
        str,
        frozenset[_NumericFact | _PeriodFact],
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
    evidence_facts = set().union(
        *(grounding_index[ref] for ref in paragraph_refs if ref in grounding_index)
    )
    paragraph_claims = _grounding_facts_from_text(paragraph.text)

    for claim in paragraph.derived_claims:
        derived_fact = _verified_derived_fact(
            claim,
            paragraph_refs,
            item_by_ref,
        )
        if derived_fact is None:
            errors.append("DERIVED_CLAIM_INVALID")
            continue
        if not _derived_claim_appears(claim, paragraph.text):
            errors.append("DERIVED_CLAIM_NOT_USED")
            continue
        evidence_facts.add(derived_fact)

    if any(
        not _grounding_claim_is_supported(claim, evidence_facts)
        for claim in paragraph_claims
    ):
        errors.append("UNGROUNDED_NUMERIC_CLAIM")
    return list(dict.fromkeys(errors))
