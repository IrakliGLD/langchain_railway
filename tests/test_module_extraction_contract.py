"""Contract test for the P0-1 module extraction (architecture-audit 2026-06-30).

Guards that:
- ``SummaryEnvelope`` lives in ``contracts.summary`` (a leaf with no LLM-hub dependency),
- ``classify_query_type`` / ``get_query_focus`` live in ``core.query_classifier``,
- ``core.llm`` still re-exports all three as the SAME objects (back-compat), so existing
  ``from core.llm import ...`` call sites and test monkeypatches keep working.

If a future refactor moves these back into ``core.llm`` or breaks the re-export, this
fails loudly instead of silently reintroducing the upward-import coupling P0-1 removed.
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import contracts.summary
import core.query_classifier
from core import llm


def test_summary_envelope_canonical_home_and_reexport():
    # Definition lives in contracts.summary; core.llm re-exports the SAME object.
    assert llm.SummaryEnvelope is contracts.summary.SummaryEnvelope
    assert contracts.summary.SummaryEnvelope.__module__ == "contracts.summary"
    assert {"answer", "claims", "citations", "confidence"} <= set(
        contracts.summary.SummaryEnvelope.model_fields
    )


def test_query_classifier_canonical_home_and_reexport():
    assert llm.classify_query_type is core.query_classifier.classify_query_type
    assert llm.get_query_focus is core.query_classifier.get_query_focus
    assert core.query_classifier.classify_query_type.__module__ == "core.query_classifier"


def test_query_classifier_behaviour_preserved():
    classify = core.query_classifier.classify_query_type
    focus = core.query_classifier.get_query_focus
    assert classify("What is the price in June 2024?") == "single_value"
    assert classify("Compare 2023 vs 2024") == "comparison"
    assert focus("balancing price") == "balancing"
    assert focus("tariff schedule") == "tariff"
