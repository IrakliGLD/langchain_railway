"""
Regression checks for Phase 5 cleanup.

Ensures deprecated modules stay removed and runtime points to new locations.
"""
import os
from pathlib import Path
from typing import Any

import sqlalchemy


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


# Ensure imports are safe in isolated test runs.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
sqlalchemy.create_engine = lambda *args, **kwargs: _DummyEngine()  # type: ignore[assignment]


def test_deprecated_files_removed():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "sql_helpers.py").exists()
    assert not (root / "prompts" / "few_shot_examples.py").exists()


def test_new_transfer_modules_present():
    root = Path(__file__).resolve().parents[1]
    assert (root / "agent" / "aggregation.py").exists()
    assert (root / "knowledge" / "sql_example_selector.py").exists()


def test_runtime_imports_use_new_paths():
    from agent import planner, sql_executor
    from core import llm

    planner_imports = planner.detect_aggregation_intent.__module__
    sql_imports = sql_executor.validate_aggregation_logic.__module__
    llm_imports = llm.get_relevant_examples.__module__

    assert planner_imports == "agent.aggregation"
    assert sql_imports == "agent.aggregation"
    assert llm_imports == "knowledge.sql_example_selector"


def test_prompt_assets_use_canonical_balancing_segment_rule():
    root = Path(__file__).resolve().parents[1]
    prompt_assets = [
        root / "core" / "llm.py",
        root / "main.py",
        root / "knowledge" / "sql_example_selector.py",
        root / "knowledge" / "sql_examples.md",
        root / "knowledge" / "balancing_price.md",
        root / "knowledge" / "market_structure.md",
        root / "analysis" / "shares.py",
    ]
    banned_snippets = [
        "segment = 'Balancing Electricity'",
        "segment='balancing_electricity'",
        "Database may contain 'Balancing Electricity', 'balancing', or other variants",
        "balancing_electricity",
    ]

    for path in prompt_assets:
        text = path.read_text(encoding="utf-8")
        for snippet in banned_snippets:
            assert snippet not in text, f"{path.name} contains stale balancing-segment guidance: {snippet}"

    assert "LOWER(REPLACE(segment, ' ', '_')) = 'balancing'" in (
        root / "knowledge" / "sql_example_selector.py"
    ).read_text(encoding="utf-8")
