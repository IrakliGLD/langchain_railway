"""An exhaustive snapshot of Standard's chart-type decision.

This is the regression net for the report charting work. The report is being
given the composition rule Standard already applies, and the mandate is that
Standard's own behaviour does not move. A hand-picked set of cases would only
pin what the author already thought of, so the input domain is enumerated in
full instead: it is finite and small enough to hold whole.

``agent/chart_pipeline._choose_chart_type`` is the entire decision surface --
explicit group type, the analyzer's preferred family, the visual-goal rule, the
shared fallback selector, and the two corrective passes after it.

**Never regenerate the golden to make a test pass.** A diff here means
Standard's charts changed, which is the one outcome this work may not produce.
Regenerate only when the change to Standard is the deliberate subject of the
commit, and say so in the message.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from itertools import combinations
from types import SimpleNamespace

from agent.chart_pipeline import _choose_chart_type
from contracts.question_analysis import ChartFamily, VisualGoal

GOLDEN_PATH = (
    Path(__file__).parent / "fixtures" / "chart_type_decision_golden.json"
)

# The complete vocabulary infer_dimension can return. Enumerating the powerset
# is the point: the combinations that break are the ones nobody lists by hand.
# Each is abbreviated to one letter in the golden keys, which is worth 250 KB
# across six thousand rows and still reads at a glance.
_DIMENSION_LETTERS = {
    "energy_qty": "e",
    "index": "i",
    "other": "o",
    "price_tariff": "p",
    "share": "s",
    "xrate": "x",
}
_DIMENSION_VOCABULARY = tuple(sorted(_DIMENSION_LETTERS))


def _dimension_code(dimensions: tuple[str, ...]) -> str:
    return "".join(_DIMENSION_LETTERS[name] for name in sorted(dimensions)) or "-"
# 8 is the pie/bar boundary in both the goal rule and the shared selector.
_CATEGORY_COUNTS = (1, 8, 9)
_VISUAL_GOALS = (None, *[goal.value for goal in VisualGoal])
# Representative dimension sets for the short-circuit sweep, where the
# semantic core is held small so the override matrix can be exhaustive.
_CORE_DIMENSION_SETS = (
    (),
    ("share",),
    ("energy_qty", "share"),
    ("price_tariff",),
)


def _dimension_powerset() -> list[tuple[str, ...]]:
    return [
        subset
        for size in range(len(_DIMENSION_VOCABULARY) + 1)
        for subset in combinations(_DIMENSION_VOCABULARY, size)
    ]


def _decide(
    *,
    dimensions: tuple[str, ...],
    goal: str | None,
    has_time: bool,
    has_categories: bool,
    category_count: int,
    group_type: str | None = None,
    explicit_user_type: str | None = None,
    family: str | None = None,
) -> str:
    group = {"type": group_type}
    if explicit_user_type is not None:
        group["_explicit_user_chart_type"] = explicit_user_type
    return _choose_chart_type(
        group=group,
        visualization=SimpleNamespace(
            preferred_chart_family=(
                None if family is None else ChartFamily(family)
            ),
            visual_goal=None if goal is None else VisualGoal(goal),
        ),
        has_time=has_time,
        has_categories=has_categories,
        # A fresh set per call: _choose_chart_type must not mutate its input,
        # and a shared set would hide it if it did.
        dimensions=set(dimensions),
        category_count=category_count,
    )


def _key(**parts: object) -> str:
    return "|".join(f"{name}={parts[name]}" for name in sorted(parts))


def build_decision_table() -> dict[str, str]:
    """Return Standard's decision for every point in the enumerated domain.

    Two suites. The semantic one sweeps the full powerset of dimensions against
    every visual goal; the override one holds the semantic core small so the
    short-circuit matrix (explicit group type, user type, preferred family) can
    itself be exhaustive.
    """

    table: dict[str, str] = {}
    for dimensions in _dimension_powerset():
        for goal in _VISUAL_GOALS:
            for has_time in (False, True):
                for has_categories in (False, True):
                    for count in _CATEGORY_COUNTS:
                        key = _key(
                            cats=int(has_categories),
                            dims=_dimension_code(dimensions),
                            goal=goal or "-",
                            n=count,
                            time=int(has_time),
                        )
                        table[key] = _decide(
                            dimensions=dimensions,
                            goal=goal,
                            has_time=has_time,
                            has_categories=has_categories,
                            category_count=count,
                        )

    for dimensions in _CORE_DIMENSION_SETS:
        for goal in (None, "composition"):
            for group_type in (None, "bar"):
                for explicit_user_type in (None, "bar"):
                    for family in (None, "pie"):
                        for has_time in (False, True):
                            for has_categories in (False, True):
                                key = _key(
                                    cats=int(has_categories),
                                    dims=_dimension_code(dimensions),
                                    family=family or "-",
                                    goal=goal or "-",
                                    group=group_type or "-",
                                    suite="override",
                                    time=int(has_time),
                                    user=explicit_user_type or "-",
                                )
                                table[key] = _decide(
                                    dimensions=dimensions,
                                    goal=goal,
                                    has_time=has_time,
                                    has_categories=has_categories,
                                    category_count=8,
                                    group_type=group_type,
                                    explicit_user_type=explicit_user_type,
                                    family=family,
                                )
    return table


def test_standard_chart_type_decisions_are_unchanged():
    """Standard's charts must not move while the report is being fixed."""

    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    actual = build_decision_table()

    assert set(actual) == set(expected), {
        "missing_from_golden": sorted(set(actual) - set(expected))[:8],
        "missing_from_actual": sorted(set(expected) - set(actual))[:8],
    }
    # Name the offenders rather than reporting a count: a bare "tables differ"
    # on six thousand rows is not diagnosable from a failure message.
    changed = {
        key: {"golden": expected[key], "now": actual[key]}
        for key in sorted(actual)
        if actual[key] != expected[key]
    }
    assert not changed, dict(list(changed.items())[:8])


def test_the_enumerated_domain_covers_every_documented_input():
    """A golden is only worth the domain behind it."""

    table = build_decision_table()
    semantic = [key for key in table if "suite=override" not in key]

    # Every dimension letter, every visual goal, and both sides of the pie
    # boundary have to appear, or the net has a hole in it.
    for letter in _DIMENSION_LETTERS.values():
        assert any(
            letter in key.split("dims=", 1)[1].split("|", 1)[0]
            for key in semantic
        ), letter
    for goal in VisualGoal:
        assert any(f"goal={goal.value}|" in key for key in semantic), goal
    for count in _CATEGORY_COUNTS:
        assert any(f"n={count}|" in key for key in semantic), count
    assert len(semantic) == (
        2 ** len(_DIMENSION_VOCABULARY)
        * len(_VISUAL_GOALS)
        * 2
        * 2
        * len(_CATEGORY_COUNTS)
    )


def test_the_decision_never_mutates_its_dimension_argument():
    """A shared set leaking a mutation would corrupt the caller's group."""

    dimensions = {"share", "energy_qty"}
    snapshot = set(dimensions)
    _choose_chart_type(
        group={"type": None},
        visualization=SimpleNamespace(
            preferred_chart_family=None,
            visual_goal=VisualGoal.COMPOSITION,
        ),
        has_time=False,
        has_categories=True,
        dimensions=dimensions,
        category_count=4,
    )

    assert dimensions == snapshot
