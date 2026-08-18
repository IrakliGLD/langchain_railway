"""Deterministic extraction of a data series the user typed into the question.

Incident 2026-08-17 (spans 0e44b3ef / c7cf532b / bc37e409). A user pasted a
twelve-month consumption profile beneath a one-line question. The pipeline had
no channel for it: ``derived_metrics`` was empty, the scenario fallback was
empty, and the summarizer prompt census carried no slot that could hold it. The
figures reached neither the statistics nor the grounding corpus, so the only
answers available were ones that ignored what the user had supplied.

This module is that channel. Extraction is deterministic on purpose -- an LLM
re-reading the paste is exactly the failure being fixed. Totals are computed
here so the model never has to add the numbers itself; a model-computed sum is
an ungrounded value that the Stage 4 provenance gate would have to repair.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("Enai")

#: A run of digits with optional thousands separators (comma, space, apostrophe)
#: and an optional decimal tail. Kept deliberately strict about the decimal
#: separator: a comma is a thousands separator in every locale this product
#: serves, so treating it as a decimal point would silently divide by 1000.
_NUMBER_RE = re.compile(r"-?\d{1,3}(?:[ ,' ]\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")

#: Georgian month names, nominative. Matched case-insensitively and by prefix so
#: declined forms ("იანვარში") still resolve.
_GEORGIAN_MONTHS = (
    "იანვარ", "თებერვალ", "მარტ", "აპრილ", "მაის", "ივნის",
    "ივლის", "აგვისტ", "სექტემბერ", "ოქტომბერ", "ნოემბერ", "დეკემბერ",
)
_ENGLISH_MONTHS = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
)
_ISO_PERIOD_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})\b")
_YEAR_RANGE = range(1990, 2100)

#: Below this, a "series" is indistinguishable from prose that happens to carry
#: numbers ("between 2024 and 2025"). Three is the smallest count that shows a
#: repeating period/value shape rather than a coincidence.
MIN_SERIES_POINTS = 3


@dataclass(frozen=True, slots=True)
class SeriesPoint:
    """One period/value pair exactly as the user supplied it."""

    period: str
    value: float


@dataclass(frozen=True, slots=True)
class UserSuppliedSeries:
    """A typed series the user typed, with aggregates computed in code."""

    points: list[SeriesPoint]
    unit: str = ""
    label: str = "user_supplied_series"
    _total: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("a user-supplied series needs at least one point")

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def total(self) -> float:
        return sum(point.value for point in self.points)

    @property
    def mean(self) -> float:
        return self.total / len(self.points)

    def as_records(self) -> list[dict[str, object]]:
        """Rows for the prompt and for the provenance corpus."""
        return [
            {
                "record_type": "user_supplied",
                "series": self.label,
                "period": point.period,
                "value": point.value,
                "unit": self.unit,
            }
            for point in self.points
        ]


def format_series_number(value: float) -> str:
    """Render a figure the way an answer will cite it.

    Never scientific notation: ``f"{4548000.0:g}"`` is ``4.548e+06``, which no
    grounding check will match against a claim that says 4,548,000, and which a
    reader would have to decode. Integral values lose their ``.0`` for the same
    reason -- the citation and the evidence have to be the same string.
    """
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _period_for_line(line: str) -> Optional[str]:
    """Resolve a line's period label, or None when the line names no period."""
    iso = _ISO_PERIOD_RE.search(line)
    if iso is not None:
        month = int(iso.group(2))
        if 1 <= month <= 12:
            return f"{iso.group(1)}-{month:02d}"
    lowered = line.lower()
    for index, stem in enumerate(_GEORGIAN_MONTHS, start=1):
        if stem in lowered:
            return f"{index:02d}"
    for index, stem in enumerate(_ENGLISH_MONTHS, start=1):
        if re.search(rf"\b{stem}", lowered):
            return f"{index:02d}"
    return None


def _value_for_line(line: str, period: str) -> Optional[float]:
    """Take the line's measurement, ignoring digits that spell the period."""
    # An ISO period contributes its own digits; remove it before reading the
    # value or "2025-01: 412000" yields 2025.
    cleaned = _ISO_PERIOD_RE.sub(" ", line) if "-" in period else line
    numbers: list[float] = []
    for match in _NUMBER_RE.finditer(cleaned):
        raw = match.group(0)
        try:
            numbers.append(float(re.sub(r"[ ,' ]", "", raw)))
        except ValueError:  # pragma: no cover - regex already constrains this
            continue
    if not numbers:
        return None
    # "January 2025 412000" names its year in full. Taking the first number
    # would report every month's consumption as 2025.
    if len(numbers) > 1 and numbers[0].is_integer() and int(numbers[0]) in _YEAR_RANGE:
        return numbers[1]
    return numbers[0]


#: A data row is essentially ``<period> <number> [unit]``. Prose that mentions a
#: month carries far more words than that, so the count of *words* left after
#: removing the period and the figures separates a pasted list from a sentence.
#: Two allows a unit and one qualifier ("412000 kWh net").
_MAX_EXTRA_WORDS = 2
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _line_is_shaped_like_a_data_row(line: str) -> bool:
    """Reject sentences that merely mention a month.

    Without this, "In January 2025 the price rose against a cold snap of 1200
    GWh demand" reads as a data point -- and reads the *year* as its value.
    Four such bullets in a research brief become a profile nobody supplied.
    """
    lowered = _ISO_PERIOD_RE.sub(" ", line).lower()
    for stem in (*_GEORGIAN_MONTHS, *_ENGLISH_MONTHS):
        if stem in lowered:
            # Remove the month word itself, including any declension tail.
            lowered = re.sub(rf"{stem}\w*", " ", lowered)
    return len(_WORD_RE.findall(lowered)) <= _MAX_EXTRA_WORDS


def extract_user_supplied_series(query: str) -> Optional[UserSuppliedSeries]:
    """Return the series a user pasted into ``query``, or None.

    Returns None rather than an empty series so callers read as "was there
    one?" instead of having to test emptiness -- the false-positive guard is
    the whole point of this function and it should be impossible to skip.
    """
    text = str(query or "")
    if not text.strip():
        return None

    points: list[SeriesPoint] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        period = _period_for_line(line)
        if period is None or period in seen:
            continue
        if not _line_is_shaped_like_a_data_row(line):
            continue
        value = _value_for_line(line, period)
        if value is None:
            continue
        seen.add(period)
        points.append(SeriesPoint(period=period, value=value))

    if len(points) < MIN_SERIES_POINTS:
        return None
    return UserSuppliedSeries(points=points)


def strip_user_supplied_series_lines(query: str) -> str:
    """Return the question with any pasted data rows removed.

    What a question ASKS FOR is decided by the question, not by figures quoted
    underneath it. On 2026-08-17 the SQL relevance guard read the Georgian word
    for consumption out of the user's own line "my consumption by months is as
    follows", concluded the question was about demand, and hard-blocked a
    correct retail-versus-wholesale query to zero rows.

    Only strips when a series is actually present, so a lone month mentioned in
    prose is never removed from the question.
    """
    text = str(query or "")
    if extract_user_supplied_series(text) is None:
        return text

    kept: list[str] = []
    for line in text.splitlines():
        period = _period_for_line(line)
        is_data_row = (
            period is not None
            and _line_is_shaped_like_a_data_row(line)
            and _value_for_line(line, period) is not None
        )
        if is_data_row:
            # The line that introduces the block belongs to it. "my consumption
            # by months is as follows:" names consumption because it is
            # labelling the figures below, not asking for consumption data --
            # and that word alone is what blocked the 2026-08-17 turn. Only a
            # colon lead-in immediately above the rows qualifies, so an actual
            # question keeps its topics.
            while kept and not kept[-1].strip():
                kept.pop()
            if kept and kept[-1].rstrip().endswith((":", "：")):
                kept.pop()
            continue
        kept.append(line)
    return "\n".join(kept)


def attach_user_supplied_series(ctx) -> bool:
    """Put the user's own figures into statistics and the grounding corpus.

    Returns True when a series was found and attached.

    Attaching to ``stats_hint`` is what makes these values citable:
    ``agent.summary_grounding._build_grounding_corpus`` reads preview,
    stats_hint and the frame, so a figure present here passes the Stage 4 gate.
    It deliberately does NOT call ``stamp_provenance`` -- that helper *replaces*
    ``provenance_cols``/``provenance_rows`` rather than appending, so stamping
    here would swap the measured frame's provenance for twelve user-typed rows
    and leave every measured figure unattributable.
    """
    series = extract_user_supplied_series(getattr(ctx, "query", ""))
    if series is None:
        return False

    records = series.as_records()
    ctx.stats_hint = (ctx.stats_hint or "") + (
        "\n\n--- USER-SUPPLIED SERIES (stated in the question, not measured) ---\n"
        f"point_count={series.point_count} "
        f"total={format_series_number(series.total)} "
        f"mean={format_series_number(series.mean)}\n"
        + json.dumps(records, default=str, indent=2)
        + "\nThese values were supplied by the user, not retrieved from the "
        "database. Use them for weighting and totals; cite them as "
        '"statistics". Do not recompute the total.'
    )
    log.info(
        "User-supplied series attached to stats_hint: points=%d total=%g",
        series.point_count,
        series.total,
    )
    return True
