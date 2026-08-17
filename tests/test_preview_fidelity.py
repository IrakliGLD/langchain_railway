"""What the data preview loses when it does not fit, and what it must not.

On the 2026-08-16 trace a 528-row retail frame reached the summarizer as 60
rows: ``rows_to_preview`` cut 528 -> 124 by dropping the middle, then the
summarizer's own compaction cut 124 -> 60. Because the rows are ordered by date
DESC and both stages keep head and tail, what survived was roughly the five most
recent months and the three oldest -- 2022, 2023, 2024 and 2025 were absent
entirely, and the question was about 2022 versus 2024.

Worse, the first stage concatenates head and tail with NO marker, so the model
sees 2026-06 followed by 2021-09 and has no way to know rows were removed.

Every behaviour here is gated on the preview being over budget. A frame that
fits is untouched.
"""

from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from analysis.stats import PREVIEW_OMISSION_MARKER, rows_to_preview

CATEGORIES = (
    ("220/380|com|other", "Commercial - other (220/380)"),
    ("220/380|com|small", "Commercial - small (220/380)"),
    ("220/380|hh|cat1", "Household cat 1, <=101 kWh (220/380)"),
    ("220/380|hh|cat2", "Household cat 2, 101-301 kWh (220/380)"),
    ("220/380|hh|cat3", "Household cat 3, >301 kWh (220/380)"),
    ("3.3-6-10|com|other", "Commercial - other (3.3-6-10)"),
    ("3.3-6-10|hh|", "Household (3.3-6-10)"),
    ("35-110|com|other", "Commercial - other (35-110)"),
)

COLS = [
    "date", "supplier", "category", "category_label",
    "distribution_tariff_gel_kwh", "supply_tariff_gel_kwh",
    "transmission_tariff_gel_kwh", "final_price_net_gel_kwh",
    "wholesale_benchmark_gel_kwh", "supply_vs_wholesale_spread_gel_kwh",
    "final_price_net_gel_mwh", "supply_company", "series_label",
]


def _retail_rows(months: int = 66):
    """The 2026-08-16 frame shape: 8 categories x N months, newest first."""
    dates = list(pd.date_range("2021-07-01", periods=months, freq="MS"))[::-1]
    rows = []
    for date in dates:
        for category, label in CATEGORIES:
            rows.append((
                date.strftime("%Y-%m-%d"), "telmico", category, label,
                0.0812, 0.1104, 0.0067, 0.1983, 0.1470, -0.0366, 198.3,
                "Telmico (Tbilisi Electricity Supply Company)",
                f"Telmico - {label}",
            ))
    return rows


def test_a_frame_that_fits_is_left_exactly_as_it_was():
    """Everything else here is gated on truncation. Small frames must not change."""
    rows = _retail_rows(months=2)
    preview = rows_to_preview(rows, COLS)

    assert PREVIEW_OMISSION_MARKER not in preview
    assert "LEGEND" not in preview
    # Every row survives, labels still inline.
    assert preview.count("Telmico (Tbilisi Electricity Supply Company)") == len(rows)


def test_a_truncated_preview_says_so():
    """Without a marker the model reads 2026-06 -> 2021-09 as contiguous."""
    preview = rows_to_preview(_retail_rows(), COLS)

    assert PREVIEW_OMISSION_MARKER in preview


def test_repeated_long_labels_become_a_legend_when_space_is_tight():
    """Three columns carried ~110 of each ~197-char row, repeated 528 times.

    ``supply_company`` (43 chars), ``series_label`` (38) and ``category_label``
    (up to 37) are each constant per key already in the row, so repeating them
    buys nothing and costs well over half the preview.
    """
    preview = rows_to_preview(_retail_rows(), COLS)

    assert "LEGEND" in preview
    # The mapping is stated, once.
    assert preview.count("Telmico (Tbilisi Electricity Supply Company)") == 1
    # The key that recovers it is still on every row.
    assert preview.count("telmico") > 10
    # Every dropped label is still recoverable from the legend.
    for _category, label in CATEGORIES:
        assert label in preview


def test_every_year_survives_truncation_of_a_multi_year_frame():
    """The defect this phase exists to fix.

    The question asked about 2022 against 2024. Both were absent from the
    prompt, along with 2023 and 2025.
    """
    preview = rows_to_preview(_retail_rows(), COLS)

    for year in ("2021", "2022", "2023", "2024", "2025", "2026"):
        assert f",{year}-" in preview or preview.count(year) > 0, year
    # Not merely present: each year keeps several months.
    for year in ("2022", "2023", "2024", "2025"):
        months = {line[:7] for line in preview.splitlines() if line.startswith(year)}
        assert len(months) >= 2, f"{year} kept only {months}"


def test_sampling_keeps_whole_dates_so_series_stay_comparable():
    """Sampling by row position would leave different categories on different
    dates, and a cross-series comparison at a given date would be impossible."""
    preview = rows_to_preview(_retail_rows(), COLS)

    per_date: dict[str, int] = {}
    for line in preview.splitlines():
        if line[:4].isdigit():
            per_date[line.split(",")[0]] = per_date.get(line.split(",")[0], 0) + 1

    assert per_date, "no data rows parsed"
    assert set(per_date.values()) == {len(CATEGORIES)}, (
        f"dates retained with a partial series set: "
        f"{sorted(d for d, n in per_date.items() if n != len(CATEGORIES))}"
    )


def test_every_year_still_survives_the_summarizer_second_compaction():
    """Audit finding, Phase 4: there are TWO compactions, and fixing one is not enough.

    ``rows_to_preview`` hands a 30,094-char preview to
    ``_compact_summarizer_preview``, which cuts it again to 12,000 by keeping
    head and tail. Measured after the first fix alone: 2021 kept 32 rows, 2022
    kept 27, 2026 kept 64 -- and 2023 and 2024 were gone, which are exactly the
    years the question was about. The second stage has to drop whole periods
    evenly too, or the first stage's work is undone downstream.
    """
    from core.llm import _compact_summarizer_preview

    compacted = _compact_summarizer_preview(rows_to_preview(_retail_rows(), COLS))

    years = {line[:4] for line in compacted.splitlines() if line[:4].isdigit()}
    assert {"2021", "2022", "2023", "2024", "2025", "2026"} <= years, sorted(years)


def test_the_second_compaction_also_keeps_whole_periods():
    """Same reason as the first stage: a partial series set at a date makes a
    cross-category comparison at that date impossible."""
    from core.llm import _compact_summarizer_preview

    compacted = _compact_summarizer_preview(rows_to_preview(_retail_rows(), COLS))

    per_date: dict[str, int] = {}
    for line in compacted.splitlines():
        if line[:4].isdigit():
            per_date[line.split(",")[0]] = per_date.get(line.split(",")[0], 0) + 1

    assert per_date
    assert set(per_date.values()) == {len(CATEGORIES)}, (
        f"partial series sets at: "
        f"{sorted(d for d, n in per_date.items() if n != len(CATEGORIES))}"
    )


def test_endpoints_are_never_dropped():
    """The newest and oldest rows anchor the range the answer may describe."""
    preview = rows_to_preview(_retail_rows(), COLS)

    assert "2026-12-01" in preview
    assert "2021-07-01" in preview
