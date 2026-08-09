"""The report's copy of Standard's composition rule must agree with Standard.

`agent/report_chart_rules` is a deliberate duplicate: Standard's chart output
must not move, so its call path is not edited. What makes the duplication safe
is this file. Every point of the frozen golden where Standard is being asked
the same question is replayed against the copy, so a divergence in *either*
direction fails a test instead of shipping.

If this file fails, one of two things happened and they need different
responses: either the copy drifted (fix the copy), or Standard changed (decide
whether the report should follow, then update both deliberately).
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from agent.report_chart_rules import (
    composition_chart_type,
    composition_chart_type_for_columns,
)
from tests.test_chart_type_decision_golden import (
    _DIMENSION_LETTERS,
    GOLDEN_PATH,
)

_LETTER_TO_DIMENSION = {
    letter: name for name, letter in _DIMENSION_LETTERS.items()
}


def _decode_dimensions(code: str) -> set[str]:
    return set() if code == "-" else {
        _LETTER_TO_DIMENSION[letter] for letter in code
    }


def _golden_composition_points():
    """Yield the golden rows that ask the report's question.

    The report always asks about a snapshot: one period, categories present,
    composition intent. Anything else in the golden is Standard answering a
    question the report never puts to it.
    """

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    for key, expected in sorted(golden.items()):
        parts = dict(part.split("=", 1) for part in key.split("|"))
        if parts.get("suite") == "override":
            continue
        if parts["goal"] != "composition":
            continue
        if parts["time"] != "0" or parts["cats"] != "1":
            continue
        yield key, _decode_dimensions(parts["dims"]), int(parts["n"]), expected


def test_the_report_rule_agrees_with_standard_on_every_golden_point():
    disagreements = {
        key: {"standard": expected, "report": actual}
        for key, dimensions, count, expected in _golden_composition_points()
        if (actual := composition_chart_type(dimensions, count)) != expected
    }

    assert not disagreements, dict(list(disagreements.items())[:8])


def test_the_compared_slice_is_not_empty_or_trivial():
    """An equivalence test over nothing passes for the wrong reason."""

    points = list(_golden_composition_points())

    assert len(points) == 2 ** len(_DIMENSION_LETTERS) * 3, len(points)
    answers = {expected for _key, _dims, _count, expected in points}
    # If Standard only ever said one thing here, agreeing with it would prove
    # nothing about the exactness rule this work exists to port.
    assert answers >= {"pie", "bar", "line"}, answers


def test_a_mixed_composition_is_not_a_pie():
    """The defect, stated directly against real column names.

    Job 26f3bbf6's pie put shares and thousand MWh in one whole. The report
    used to answer pie here because the fallback selector tests membership.
    """

    assert composition_chart_type_for_columns(
        ["share_hydro", "share_thermal", "share_wind"], 3
    ) == "pie"
    assert composition_chart_type_for_columns(
        ["share_hydro", "quantity_hydro"], 2
    ) == "bar"


@pytest.mark.parametrize(
    ("dimensions", "count", "expected"),
    [
        # Exactness, from both sides.
        ({"share"}, 8, "pie"),
        ({"share"}, 9, "bar"),
        ({"share", "energy_qty"}, 4, "bar"),
        ({"share", "other"}, 4, "bar"),
        # A continuous measure with no share to anchor it is a line, never a
        # slice -- Standard's corrective pass, which the report never ran.
        ({"price_tariff"}, 4, "line"),
        ({"xrate"}, 4, "line"),
        ({"price_tariff", "energy_qty"}, 4, "line"),
        # ...but share present keeps it out of the corrective.
        ({"price_tariff", "share"}, 4, "bar"),
        # Nothing to plot at all still answers, rather than raising.
        (set(), 4, "bar"),
    ],
)
def test_named_rules_hold(dimensions, count, expected):
    assert composition_chart_type(dimensions, count) == expected
