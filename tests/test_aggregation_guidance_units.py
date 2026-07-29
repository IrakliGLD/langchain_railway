"""Aggregate SQL aliases must preserve the unit of the column they aggregate.

The report grounding validator refuses any claim on a column whose unit it
cannot resolve, and unit resolution works from the column name. Guidance that
tells the SQL writer to alias an aggregate to a bare word ("AS total") throws
that information away, so every number in that column becomes unclaimable and
no repair can recover it -- the failure mode of report job 1779c440.
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from agent.aggregation import get_aggregation_guidance
from agent.report_evidence import _inferred_unit_by_column

_ALIAS_PATTERN = re.compile(r"\bAS\s+([a-z_][a-z0-9_]*)", re.IGNORECASE)

# Aliases that name a category or a period rather than a magnitude. A claim is
# never made against these, so they need no unit.
_LABEL_ALIASES = {
    "segment",
    "type_tech",
    "entity",
    "period",
    "date",
    "category",
}


def _aliases(guidance: str) -> set[str]:
    return {alias.lower() for alias in _ALIAS_PATTERN.findall(guidance)}


@pytest.mark.parametrize(
    ("intent", "aggregate"),
    [
        ({"needs_total": True}, "SUM"),
        ({"needs_average": True}, "AVG"),
        ({"needs_breakdown": True}, "SUM"),
    ],
)
def test_concrete_example_aliases_resolve_to_a_unit(
    intent: dict,
    aggregate: str,
):
    guidance = get_aggregation_guidance(intent)

    for alias in _aliases(guidance):
        # "total_column_name" is the instruction's placeholder, not an emitted
        # column; the shape of that template is pinned by the test below.
        if "column_name" in alias or alias in _LABEL_ALIASES:
            continue
        units = _inferred_unit_by_column([alias])
        assert alias in units, (
            f"{aggregate} guidance shows the example alias 'AS {alias}', which "
            f"resolves to no unit. A report claim on that column cannot be "
            f"verified."
        )


@pytest.mark.parametrize(
    ("intent", "bare_alias"),
    [
        ({"needs_total": True}, "total"),
        ({"needs_average": True}, "average"),
        ({"needs_breakdown": True}, "sum"),
    ],
)
def test_guidance_never_prescribes_a_bare_aggregate_alias(
    intent: dict,
    bare_alias: str,
):
    guidance = get_aggregation_guidance(intent)

    assert bare_alias not in _aliases(guidance), (
        f"Guidance prescribes the bare alias 'AS {bare_alias}', which discards "
        "the aggregated column's unit."
    )
