"""The preview must show both ends of the date range, not just the start.

``rows_to_preview`` applies a row slice BEFORE its character budget. The row
cap was a hardcoded 200, so on a 1,056-row retail frame the model received the
first 200 rows and no tail -- roughly one year of a five-year question -- while
the character cap sat unused at ~8,000 of 18,000. The character cap is the one
that truncates intelligently, dropping middle rows and keeping the first and
last, so it is the one that should bind.
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from analysis.stats import rows_to_preview  # noqa: E402
from config import PREVIEW_MAX_CHARS, PREVIEW_MAX_ROWS  # noqa: E402


def _retail_rows(months=66, suppliers=2, categories=8):
    """A frame the size the retail tool actually returns."""
    rows = []
    for m in range(months):
        for s in range(suppliers):
            for c in range(categories):
                rows.append(
                    (
                        f"{2021 + m // 12}-{m % 12 + 1:02d}-01",
                        f"supplier_{s}",
                        f"cat_{c}",
                        0.19 + 0.0005 * m,
                    )
                )
    return rows


_COLS = ["date", "supplier", "category", "final_price_net_gel_kwh"]


def test_the_last_row_survives_a_full_size_retail_frame():
    """The tail is what a trend question needs and what the row cap removed."""
    rows = _retail_rows()
    assert len(rows) == 1056

    preview = rows_to_preview(rows, _COLS)
    lines = preview.strip().splitlines()

    # Located by content, not by index: a truncated preview may now open with a
    # legend and/or an omission note before the CSV header, so the first data
    # row is no longer always line 1. What matters is that it is still there.
    assert any(line.startswith(rows[0][0]) for line in lines), "first row missing"
    assert lines[-1].startswith(rows[-1][0]), (
        "last row missing: the preview still ends where the row cap cut it, so "
        "a trend question cannot see where the series ended"
    )


def test_the_character_budget_is_what_binds_not_the_row_count():
    rows = _retail_rows()
    preview = rows_to_preview(rows, _COLS)

    assert len(preview) <= PREVIEW_MAX_CHARS
    # More than the old hardcoded slice, so the extra budget is really used.
    assert len(preview.strip().splitlines()) - 1 > 200


def test_the_caps_are_configurable_rather_than_hardcoded():
    assert PREVIEW_MAX_ROWS > 200
    assert PREVIEW_MAX_CHARS >= 18_000


def test_explicit_arguments_still_win():
    """Callers that pass a cap must keep getting exactly that cap."""
    rows = _retail_rows()
    preview = rows_to_preview(rows, _COLS, max_rows=10)
    assert len(preview.strip().splitlines()) - 1 == 10


def test_a_small_result_is_returned_whole():
    rows = _retail_rows(months=2, suppliers=1, categories=2)
    preview = rows_to_preview(rows, _COLS)
    assert len(preview.strip().splitlines()) - 1 == len(rows)
