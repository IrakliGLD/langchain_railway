"""
Unit tests for _generate_why_summary_payload wording quality.
Tests the three bugs fixed:
  1. Distinguishes missing share data from no-shift
  2. xrate contradiction when mix_pressure == 0
  3. Correct observational wording
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Stub all heavy dependencies before importing anything from the project
for key in ["SUPABASE_DB_URL", "GOOGLE_API_KEY", "ENAI_GATEWAY_SECRET",
            "ENAI_SESSION_SIGNING_SECRET", "ENAI_EVALUATE_SECRET"]:
    os.environ.setdefault(key, "test_stub")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock heavy modules that would cause import failures
_mock_modules = [
    "google.generativeai",
    "google.generativeai.types",
    "google.generativeai.types.content_types",
    "sqlalchemy",
    "sqlalchemy.exc",
    "sqlalchemy.engine",
    "sqlalchemy.pool",
]
for mod in _mock_modules:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Provide a mock 'text' function for sqlalchemy
sys.modules["sqlalchemy"].text = MagicMock()

# Now mock the core modules that require DB
sys.modules["core"] = MagicMock()
sys.modules["core.query_executor"] = MagicMock()
sys.modules["core.sql_generator"] = MagicMock()

# Mock analysis modules
sys.modules["analysis"] = MagicMock()
sys.modules["analysis.stats"] = MagicMock()
sys.modules["analysis.seasonal_stats"] = MagicMock()
sys.modules["analysis.shares"] = MagicMock()

# Mock agent sub-modules that need DB
sys.modules["agent.provenance"] = MagicMock()
sys.modules["agent.sql_executor"] = MagicMock()

# Mock utils
sys.modules["utils"] = MagicMock()
sys.modules["utils.trace_logging"] = MagicMock()

# Mock models
sys.modules["models"] = MagicMock()

# Now we can import the function
# We need to import config first since analyzer uses it
try:
    import config
except Exception:
    sys.modules["config"] = MagicMock()

# Import context stubs
try:
    import context
except Exception:
    sys.modules["context"] = MagicMock()

# Finally import the function under test
from agent.analyzer import _generate_why_summary_payload


class TestNoShareData(unittest.TestCase):
    """Bug 1: has_share_evidence=False says 'not available', not 'no shift'."""

    def test_missing_share_says_unavailable(self):
        summary, claims = _generate_why_summary_payload(
            cur_gel=149.3, prev_gel=150.0,
            cur_xrate=2.71, prev_xrate=2.70,
            cur_shares={}, prev_shares={},
            has_share_evidence=False,
        )
        self.assertIsNotNone(summary)
        self.assertIn("cannot be assessed", summary.lower())
        self.assertNotIn("do not show a large mix shift", summary)

    def test_genuine_no_shift_says_no_shift(self):
        summary, claims = _generate_why_summary_payload(
            cur_gel=149.3, prev_gel=150.0,
            cur_xrate=2.71, prev_xrate=2.70,
            cur_shares={"share_import": 0.10},
            prev_shares={"share_import": 0.10},
            has_share_evidence=True,
        )
        self.assertIsNotNone(summary)
        self.assertIn("do not show a large mix shift", summary)


class TestXrateContradiction(unittest.TestCase):
    """Bug 2: When xrate direction != price direction."""

    def test_xrate_up_price_down(self):
        """Nov 2023: price dropped 150->149.3, xrate rose 2.70->2.71."""
        summary, _ = _generate_why_summary_payload(
            cur_gel=149.3, prev_gel=150.0,
            cur_xrate=2.71, prev_xrate=2.70,
            cur_shares={}, prev_shares={},
            has_share_evidence=True,
        )
        self.assertIn("opposite direction", summary.lower())

    def test_xrate_down_price_up(self):
        summary, _ = _generate_why_summary_payload(
            cur_gel=155.0, prev_gel=150.0,
            cur_xrate=2.65, prev_xrate=2.70,
            cur_shares={}, prev_shares={},
            has_share_evidence=True,
        )
        self.assertIn("opposite direction", summary.lower())

    def test_no_contradiction_when_aligned(self):
        """Both rise -> no contradiction."""
        summary, _ = _generate_why_summary_payload(
            cur_gel=155.0, prev_gel=150.0,
            cur_xrate=2.75, prev_xrate=2.70,
            cur_shares={}, prev_shares={},
            has_share_evidence=True,
        )
        self.assertNotIn("opposite direction", summary.lower())


class TestNoSignals(unittest.TestCase):
    """When both mix and xrate are neutral."""

    def test_no_signals_at_all(self):
        summary, _ = _generate_why_summary_payload(
            cur_gel=149.3, prev_gel=150.0,
            cur_xrate=2.70, prev_xrate=2.70,
            cur_shares={}, prev_shares={},
            has_share_evidence=True,
        )
        self.assertIn("no single dominant", summary.lower())


class TestBothContradict(unittest.TestCase):
    """When both mix and xrate oppose price direction."""

    def test_both_contradict_price(self):
        """Price rose, but mix is downward and xrate fell."""
        summary, _ = _generate_why_summary_payload(
            cur_gel=155.0, prev_gel=150.0,
            cur_xrate=2.65, prev_xrate=2.70,
            # Cheap source gained -> downward mix pressure
            cur_shares={"share_regulated_hpp": 0.60},
            prev_shares={"share_regulated_hpp": 0.40},
            has_share_evidence=True,
        )
        self.assertIn("do not fully explain", summary.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
