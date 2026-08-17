"""Scope that is partly stated should be partly used, not thrown away.

``_compose_category`` needs a voltage AND a consumer class; given one it returns
None and the tool widens to all eight categories. On the 2026-08-16 trace the
question named a voltage ("6-10") and no class, so a two-category question was
answered across eight -- four of them households, for a customer who plainly
was not one.

The same test file covers the unit trap that sits underneath it: "6-10" is a
**kV** alias, and the question said `6-10 კვტ` -- kilowatts, a connection power.
A 6-10 kW customer is a 220/380 V customer. Only the missing class word stopped
that becoming a confidently wrong voltage.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from agent.tools.end_user_price_tools import (
    END_USER_CATEGORIES,
    resolve_scope,
    resolve_voltage,
)
from tests.test_end_user_price_tool import _stub_rows


class TestVoltageIsOnlyReadFromAVoltage:
    """A number next to a power unit is not a voltage."""

    def test_a_stated_kilovolt_level_resolves(self):
        assert resolve_voltage("telmico, 6-10 kv, commercial") == "3.3-6-10"
        assert resolve_voltage("35-110 kv") == "35-110"
        assert resolve_voltage("220/380 v") == "220/380"

    def test_kilowatts_are_not_kilovolts(self):
        """The 2026-08-16 question: `6-10 კვტ` is 6-10 kW of connection power.

        A customer with a 6-10 kW connection is served at 220/380 V, so reading
        it as the 3.3-6-10 kV level names the wrong stack entirely.
        """
        assert resolve_voltage("თელმიკო, 6-10 კვტ მომხმარებელი") is None
        assert resolve_voltage("telmico, 6-10 kw customer") is None
        assert resolve_voltage("6-10 kva") is None

    def test_en_dashes_resolve_like_hyphens(self):
        """The analyzer emitted `6–10`; only the Georgian original's plain
        hyphen matched, which is luck rather than handling."""
        assert resolve_voltage("6–10 kv") == "3.3-6-10"
        assert resolve_voltage("35–110 kv") == "35-110"

    def test_a_power_rating_does_not_pin_a_category_either(self):
        """The whole point: no silently-wrong category from a kW figure."""
        _supplier, category = resolve_scope("telmico 6-10 kw commercial customer")
        assert category != "3.3-6-10|com|other"


class TestPartialScopeNarrowsInsteadOfWidening:
    def test_a_voltage_without_a_class_narrows_to_that_voltage(self, monkeypatch):
        import agent.tools.end_user_price_tools as tool_module

        captured = {}

        def _capture(sql, params=None):
            captured["params"] = params or {}
            return _stub_rows()(sql, params)

        monkeypatch.setattr(tool_module, "run_text_query", _capture)
        tool_module.get_end_user_prices(supplier="telmico", voltage="3.3-6-10")

        bound = {
            value for key, value in captured["params"].items()
            if key.startswith("volate_")
        }
        assert bound == {"3.3-6-10"}, bound
        # Exactly the two categories published at that voltage, not all eight.
        expected = sum(1 for c in END_USER_CATEGORIES if c.volate == "3.3-6-10")
        assert expected == 2
        cat_ids = {
            value for key, value in captured["params"].items()
            if key.startswith("cat_id_")
        }
        assert len(cat_ids) == 2, cat_ids

    def test_naming_neither_still_widens_to_all_eight(self, monkeypatch):
        import agent.tools.end_user_price_tools as tool_module

        captured = {}

        def _capture(sql, params=None):
            captured["params"] = params or {}
            return _stub_rows()(sql, params)

        monkeypatch.setattr(tool_module, "run_text_query", _capture)
        tool_module.get_end_user_prices(supplier="telmico")

        cat_ids = {
            value for key, value in captured["params"].items()
            if key.startswith("cat_id_")
        }
        assert len(cat_ids) == 8

    def test_an_unknown_voltage_raises_rather_than_guessing(self):
        import agent.tools.end_user_price_tools as tool_module

        with pytest.raises(ValueError, match="voltage"):
            tool_module.get_end_user_prices(voltage="400/230")

    def test_voltage_and_class_together_still_pin_one_category(self):
        _supplier, category = resolve_scope("telmico, 6-10 kv, commercial")
        assert category == "3.3-6-10|com|other"


def test_the_planner_passes_a_voltage_when_the_category_is_unresolved():
    """Wiring: the tool gaining a parameter nothing sets changes nothing.

    The planner fills tool params from the same haystack the clarify gate reads,
    so a voltage it can see must reach the tool.
    """
    from agent.planner import build_end_user_price_params

    params = build_end_user_price_params("telmico customer at 6-10 kv")

    assert params.get("supplier") == "telmico"
    assert params.get("voltage") == "3.3-6-10"
    assert "category" not in params

    # A fully-scoped question still pins the category and needs no voltage.
    pinned = build_end_user_price_params("telmico, 6-10 kv, commercial")
    assert pinned.get("category") == "3.3-6-10|com|other"
    assert "voltage" not in pinned
