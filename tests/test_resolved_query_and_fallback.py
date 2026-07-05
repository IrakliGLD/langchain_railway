"""
Tests for resolved-query propagation, entity normalization, and
context-preserving fallback (Phases 1–4 of the analyzer-coercion fix).
"""
import os
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import sqlalchemy

# Ensure config validation passes before importing project modules.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


class _DummyResult:
    def fetchall(self):
        return []

    def keys(self):
        return []


class _DummyConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args: Any, **kwargs: Any):
        return _DummyResult()


class _DummyEngine:
    def connect(self):
        return _DummyConnection()


sqlalchemy.create_engine = lambda *args, **kwargs: _DummyEngine()  # type: ignore[assignment]

from agent.planner import normalize_balancing_entities  # noqa: E402
from agent.router import extract_balancing_entities  # noqa: E402
from agent.tool_adapter import ToolExecutionResult  # noqa: E402
from agent.tools.composition_tools import ALLOWED_BALANCING_ENTITIES  # noqa: E402
from models import QueryContext  # noqa: E402

# ---------------------------------------------------------------------------
# Phase 2: normalize_balancing_entities
# ---------------------------------------------------------------------------


def test_normalize_passthrough_valid_entity():
    """Valid entities pass through unchanged."""
    assert normalize_balancing_entities(["import"]) == ["import"]


def test_normalize_underscore_form():
    """Space-separated form is normalized to underscore."""
    result = normalize_balancing_entities(["deregulated hydro"])
    assert "deregulated_hydro" in result


def test_normalize_cheap_energy_expands_to_cheap_tier():
    """'cheap energy' expands to all cheap-tier entities."""
    result = normalize_balancing_entities(["cheap energy"])
    assert "regulated_hpp" in result
    assert "deregulated_hydro" in result
    # Should not include expensive entities
    assert "thermal_ppa" not in result
    assert "import" not in result


def test_normalize_expensive_expands_to_expensive_tier():
    """'expensive sources' expands to expensive-tier entities."""
    result = normalize_balancing_entities(["expensive sources"])
    assert "thermal_ppa" in result
    assert "import" in result
    assert "regulated_new_tpp" in result
    # Should not include cheap entities
    assert "regulated_hpp" not in result


def test_normalize_mixed_valid_and_semantic():
    """Mix of valid entity and semantic concept resolves correctly."""
    result = normalize_balancing_entities(["import", "cheap energy"])
    assert "import" in result
    assert "regulated_hpp" in result
    assert "deregulated_hydro" in result


def test_normalize_mixed_valid_and_unresolved_returns_none():
    """Any unresolved entity should fail closed instead of partially executing."""
    result = normalize_balancing_entities(["import", "completely_invalid_xyz"])
    assert result is None


def test_normalize_label_substring_match():
    """Label-based resolution works for partial matches."""
    result = normalize_balancing_entities(["hydro"])
    assert len(result) > 0
    # Should match at least deregulated_hydro via label "deregulated hydro"
    assert "deregulated_hydro" in result


def test_normalize_nonsense_returns_none():
    """Unresolvable entities signal unresolved_concept via None."""
    result = normalize_balancing_entities(["completely_invalid_xyz"])
    assert result is None


def test_normalize_empty_input():
    """Empty input returns empty list (tool default: all entities)."""
    assert normalize_balancing_entities([]) == []


def test_normalize_preserves_allowed_order():
    """Output order matches ALLOWED_BALANCING_ENTITIES order."""
    result = normalize_balancing_entities(["thermal_ppa", "import"])
    allowed_list = [e for e in ALLOWED_BALANCING_ENTITIES if e.lower() in {r.lower() for r in result}]
    assert result == allowed_list


def test_normalize_generation_aliases_from_analyzer_labels():
    """Analyzer-facing plural labels should normalize to tool entities."""
    result = normalize_balancing_entities(
        ["old regulated TPPs", "imports", "renewable PPAs", "Thermal Generation PPAs"]
    )
    assert result == ["import", "regulated_old_tpp", "renewable_ppa", "thermal_ppa"]


def test_normalize_regulated_thermal_alias_expands_to_both_tpp_groups():
    result = normalize_balancing_entities(["all regulated thermal"])
    assert result == ["regulated_new_tpp", "regulated_old_tpp"]


def test_normalize_residual_bucket_alias_expands_to_canonical_residual_components():
    result = normalize_balancing_entities(["ppa_cfd_import_residual"])
    assert result == ["import", "renewable_ppa", "thermal_ppa", "CfD_scheme"]


def test_extract_balancing_entities_supports_full_eight_component_vocabulary():
    result = extract_balancing_entities(
        "aggregated share of renewable ppa, thermal ppa, cfd_scheme, regulated hydro, "
        "all regulated thermal and deregulated hydro in balancing electricity"
    )
    assert result == [
        "renewable_ppa",
        "thermal_ppa",
        "CfD_scheme",
        "regulated_hpp",
        "regulated_new_tpp",
        "regulated_old_tpp",
        "deregulated_hydro",
    ]


def test_extract_balancing_entities_does_not_match_regulated_inside_deregulated():
    result = extract_balancing_entities("deregulated hydro")
    assert result == ["deregulated_hydro"]


def test_extract_balancing_entities_does_not_match_old_group_inside_new_regulated_tpp():
    result = extract_balancing_entities("new regulated tpp")
    assert result == ["regulated_new_tpp"]


# ---------------------------------------------------------------------------
# Phase 1: resolved_query is set on ctx
# ---------------------------------------------------------------------------


def test_resolved_query_field_defaults_empty():
    """resolved_query defaults to empty string on fresh QueryContext."""
    ctx = QueryContext(query="test")
    assert ctx.resolved_query == ""
