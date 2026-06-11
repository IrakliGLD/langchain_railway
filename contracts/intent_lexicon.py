"""
Multilingual intent keyword lexicon — single source of truth (A3, 2026-06-10).

The same English/Georgian/Russian keyword lists were hand-maintained in
several modules and had already drifted. This module co-locates them, one
named constant per concept, languages together. Consumers import the
constants; the matching logic (regex vs ``any(... in ...)``) stays at the
call sites.

RULES:
- Word lists only — no matching logic here.
- Migration moved each list VERBATIM; do not "harmonize" two lists that look
  similar without checking every consumer (e.g. ``MODE_ANALYTICAL_KEYWORDS``
  vs the regex alternation in ``utils/query_validation._ANALYTICAL_KEYWORDS``
  cover the same concept but have different content and different consumers —
  converging them is a routing behavior change, not a refactor).

Remaining consumers to migrate (A3.d follow-up):
- ``utils/query_validation.py`` — ``_ANALYTICAL_KEYWORDS`` regex,
  ``_DATA_INTENT_PATTERN``, definition/regulation pattern lists, topic maps.
- ``agent/router.py`` — tool keyword/semantic term tables.
"""

# --- Market-side narrowing (consumer: agent/sql_executor.py) ----------------
# type_tech side-narrowing intent. The legacy SQL path post-filters a
# mixed-side generation result down to ONE market side only when the question
# explicitly signals that side (audit L1: no silent supply default).
DEMAND_SIDE_KEYWORDS = ("demand", "consumption", "loss", "export")
SUPPLY_SIDE_KEYWORDS = (
    # English
    "generation", "generate", "generated", "generating",
    "supply", "supplied", "produce", "produced", "production", "output",
    # Georgian
    "გენერაცია", "წარმოება", "გამომუშავება",
    # Russian
    "генерация", "выработка", "производство",
)

# --- Analyst-mode selection (consumer: agent/planner.py) --------------------
# Presence of any of these switches the pipeline to "analyst" mode.
# NOTE: distinct from utils/query_validation's _ANALYTICAL_KEYWORDS regex
# (different content, different consumer) — see module docstring.
MODE_ANALYTICAL_KEYWORDS = {
    "trend", "change", "growth", "increase", "decrease", "compare", "impact",
    "volatility", "pattern", "season", "relationship", "correlation", "evolution",
    "driver", "cause", "effect", "factor", "reason", "influence", "depend", "why", "behind",
    "payoff", "hypothetical", "scenario",
}

# --- Share/composition intent (consumer: models.QueryContext) ---------------
SHARE_INTENT_QUERY_SIGNALS = ("share", "composition", "contribute", "contribution")
SHARE_INTENT_CLASSIFIER_SIGNALS = ("share", "composition")
# Signals that a composition dataset is being used for a custom price/quantity
# calculation rather than a share answer (suppresses the share override).
CUSTOM_PRICE_CALC_SIGNALS = ("weighted average", "average price", "weighted avg", "mean price")
QUANTITY_REQUEST_SIGNALS = ("how much", "quantity", "volume", "mwh")
