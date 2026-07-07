"""Fixture-contract tests for the routing golden set (§5.3).

The golden-set RUNNER (evaluation/routing_golden_set.py) calls a live LLM and
is deliberately NOT exercised here — tests/ never makes real LLM calls. These
tests pin the fixture file's structure and enum validity so a bad fixture edit
fails in CI rather than at eval time.
"""

from __future__ import annotations

import json
import os

# Ensure config validation passes if any transitive import pulls config.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from evaluation.routing_golden_set import (
    _ASSERTABLE_FIELDS,
    load_fixtures,
    main,
    validate_expected_enums,
)


def test_fixtures_load_and_pass_enum_validation():
    cases = load_fixtures()
    assert len(cases) >= 10
    validate_expected_enums(cases)


def test_fixture_ids_unique_and_expected_fields_known():
    cases = load_fixtures()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert case["query"].strip()
        assert set(case["expected"]) <= set(_ASSERTABLE_FIELDS)


def test_unknown_expected_field_rejected(tmp_path):
    bad = {"version": 1, "cases": [{"id": "x", "query": "q", "expected": {"nope": "y"}}]}
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown expected fields"):
        load_fixtures(path)


def test_duplicate_id_rejected(tmp_path):
    bad = {
        "version": 1,
        "cases": [
            {"id": "x", "query": "q1", "expected": {"answer_kind": "scalar"}},
            {"id": "x", "query": "q2", "expected": {"answer_kind": "scalar"}},
        ],
    }
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        load_fixtures(path)


def test_invalid_enum_value_rejected():
    cases = [{"id": "x", "query": "q", "expected": {"answer_kind": "not_a_kind"}}]
    with pytest.raises(ValueError, match="not a valid answer_kind"):
        validate_expected_enums(cases)


def test_dry_run_needs_no_llm_and_exits_zero():
    assert main(["--dry-run"]) == 0
