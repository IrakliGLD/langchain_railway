"""The comparison template must have room for a multi-period comparison.

2026-08-17: a make-or-buy answer covered only 2026 while its evidence carried six
years and said so. The model was not choosing to drop 2021-2025 -- it was out of
budget. Both runs landed at roughly 315 and 360 words against a template ceiling
of 400, i.e. at the cap.

The cap was also out of line with its siblings: comparison 150-400 sat BELOW
forecast (200-500) and regulatory_procedure (200-600), and at half of
data_explanation (300-800), although a comparison carrying per-period figures
plus caveats is at least as demanding as an explanation.

Two things are needed: room, and an instruction to use it on every period the
evidence holds rather than only the most recent.
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

from skills.loader import load_reference

_TEMPLATES = "answer-templates.md"


def _section(name: str) -> str:
    text = load_reference("answer-composer", _TEMPLATES)
    assert text, "answer-templates.md did not load"
    start = text.index(f"## Template: {name}")
    rest = text[start + 1:]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


def _budget(name: str) -> tuple[int, int]:
    match = re.search(r"\*\*Length\*\*:\s*(\d+)\s*-\s*(\d+)\s*words", _section(name))
    assert match, f"no word budget found for template {name!r}"
    return int(match.group(1)), int(match.group(2))


def test_a_comparison_may_run_as_long_as_an_explanation():
    """A per-period comparison is at least as demanding as a driver analysis."""
    _, comparison_max = _budget("comparison")
    _, explanation_max = _budget("data_explanation / driver analysis")

    assert comparison_max >= explanation_max, (
        f"comparison still capped at {comparison_max} against "
        f"{explanation_max} for an explanation"
    )


def test_the_comparison_ceiling_clears_what_the_answers_were_hitting():
    """Both 2026-08-17 answers landed at ~315-360 words against a 400 ceiling."""
    _, comparison_max = _budget("comparison")

    assert comparison_max > 400


def test_the_floor_still_allows_a_short_two_value_comparison():
    """"Compare summer and winter prices" must not be padded to fill the range."""
    comparison_min, _ = _budget("comparison")

    assert comparison_min <= 200


def test_the_template_asks_for_every_period_the_evidence_carries():
    """The structure said "over what period", singular, and never asked for more.

    Removing the cap alone leaves the model free to keep reporting only the
    latest period.
    """
    section = _section("comparison").lower()

    assert "periods" in section
    assert "each period" in section or "every period" in section
    # And says not to stop at the newest one, which is what happened.
    assert "most recent" in section or "latest" in section


class TestNothingElseMoved:
    def test_the_other_template_budgets_are_unchanged(self):
        assert _budget("factual_lookup / single_value") == (50, 150)
        assert _budget("data_retrieval / list") == (100, 300)
        assert _budget("data_explanation / driver analysis") == (300, 800)
        assert _budget("forecast") == (200, 500)
        assert _budget("conceptual_definition") == (100, 300)

    def test_brief_mode_still_caps_hard(self):
        """A longer template must not loosen the user-selected brief contract."""
        modes = load_reference("answer-composer", "answer-length-modes.md")

        assert "Hard maximum: 140 words" in modes
