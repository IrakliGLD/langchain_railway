"""``tariff_context`` must not plan the generation-side tariff tool on a retail frame.

Every retail run on record that planned it fetched **0 rows**:

    Evidence loop: fetched tariff_context via get_tariffs (0 rows)

and the same runs logged

    Plan validation: render_style=DETERMINISTIC but plan has 1
    narrative-augmentation step(s) (['tariff_context'])

``tariff_context`` means "the regulated tariff series, as context". On a
wholesale-price question that is `get_tariffs` -- what a regulated PLANT is paid.
On a retail question the primary data already IS the tariff series, in GEL/kWh,
and `get_tariffs` is a different dataset in GEL/MWh. Planning it adds a round
trip, an empty frame, and a validator warning, and answers nothing.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.evidence_planner import _role_to_default_tool
from contracts.question_analysis import EvidenceRole, ToolName


def test_a_retail_primary_plans_no_tariff_context_tool():
    """The retail frame already carries the tariff series it would be fetching."""
    mapping = _role_to_default_tool(ToolName.GET_END_USER_PRICES.value)

    assert EvidenceRole.TARIFF_CONTEXT.value not in mapping


def test_a_wholesale_primary_still_maps_tariff_context_to_get_tariffs():
    """The role is meaningful there and must keep working."""
    mapping = _role_to_default_tool(ToolName.GET_PRICES.value)

    assert mapping[EvidenceRole.TARIFF_CONTEXT.value] == ToolName.GET_TARIFFS.value


def test_the_other_roles_are_unchanged_on_a_retail_primary():
    """Narrow change: only tariff_context is affected."""
    mapping = _role_to_default_tool(ToolName.GET_END_USER_PRICES.value)

    assert mapping[EvidenceRole.COMPOSITION_CONTEXT.value] == (
        ToolName.GET_BALANCING_COMPOSITION.value
    )
    assert mapping[EvidenceRole.CORRELATION_DRIVER.value] == ToolName.GET_PRICES.value


def test_the_prices_primary_correlation_override_still_applies():
    """Pre-existing behaviour that shares this function."""
    mapping = _role_to_default_tool(ToolName.GET_PRICES.value)

    assert mapping[EvidenceRole.CORRELATION_DRIVER.value] == (
        ToolName.GET_BALANCING_COMPOSITION.value
    )


def test_an_unrelated_primary_keeps_the_base_mapping():
    mapping = _role_to_default_tool(ToolName.GET_GENERATION_MIX.value)

    assert mapping[EvidenceRole.TARIFF_CONTEXT.value] == ToolName.GET_TARIFFS.value
