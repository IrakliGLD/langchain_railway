"""Signals for deterministic residual-price calculations.

Single semantic authority for the "negligible unknown-price residual" constraint.

The balancing price is a weighted average over buckets whose prices are known
(regulated thermal/hydro, deregulated renewables) and buckets whose prices are
not (import, PPA, CfD). The implied PPA/CfD price is only recoverable for months
where the *unknown* layer is small enough that the residual is dominated by
PPA/CfD rather than import. Users state that same constraint from either side:

    uncovered/complement side : "months where import share is less than 0.5%"
    covered/coverage side     : "months where ppa + cfd + regulated + deregulated
                                 combined are more than 99.5%"

These are the same condition (complements to 100%). Both normalize here to ONE
quantity — ``max_uncovered_share`` — so the routing gate and the downstream
month filter cannot disagree. Encoding only one phrasing is what sent the
coverage-side wording to ``ambiguous``/``clarify`` (prod trace span-28451264,
2026-07-25) and would ALSO have filtered ``share_import > 0.995`` (zero months)
had it routed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Above this fraction the unknown layer is no longer negligible, so the
# deterministic approximation is not defensible and must not be routed.
NEGLIGIBLE_UNCOVERED_SHARE_MAX = 0.01

# Intents served by the deterministic residual-price answer. Single authority:
# the summarizer authorizes its shortcut on these, and the pipeline uses them to
# decide that balancing driver-context enrichment is required — otherwise an
# analyzer-emitted intent would arrive with none of the columns the calculation
# needs (``residual_contribution_*``, ``known_price_coverage_ok``) and silently
# degrade to a narrative answer.
RESIDUAL_DIRECT_INTENTS = frozenset({
    "residual_weighted_price_calculation",
    "residual_weighted_price_followup",
    "implied_ppa_cfd_price_approximation",
})


def has_residual_direct_intent(ctx: object) -> bool:
    """True when an authoritative analysis requests a residual-price answer."""
    if not bool(getattr(ctx, "has_authoritative_question_analysis", False)):
        return False
    qa = getattr(ctx, "question_analysis", None)
    if qa is None:
        return False
    intent = str(
        getattr(getattr(qa, "classification", None), "intent", "") or ""
    ).strip().lower()
    return intent in RESIDUAL_DIRECT_INTENTS

_UNCOVERED_THRESHOLD_RE = re.compile(
    r"(?:less than|below|under|at most|no more than|fewer than|maximum of)\s+"
    r"(?P<value>\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)

_COVERED_THRESHOLD_RE = re.compile(
    r"(?:more than|greater than|above|over|exceed(?:ed|s|ing)?|at least|not less than|minimum of)\s+"
    r"(?P<value>\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)

_THRESHOLD_CLAUSE_BOUNDARY_RE = re.compile(
    r"[.;!?]|\b(?:but|when|where|while)\b",
    re.IGNORECASE,
)

# Item 2: the price intent must not hinge on one exact phrase.
_PRICE_INTENT_PHRASES = (
    "weighted average",
    "average price",
    "weighted avg",
    "avg price",
    "mean price",
    "estimate the price",
    "estimate price",
    "implied price",
)


@dataclass(frozen=True)
class ResidualCoverageThreshold:
    """Normalized negligible-residual constraint.

    ``max_uncovered_share`` is the largest allowed share (as a fraction in
    ``[0, 1]``) of the unknown-price layer, regardless of which side the user
    phrased it from. ``framing`` records the original side for message wording.
    """

    max_uncovered_share: float
    framing: str  # "uncovered" | "covered"
    phrase: str


def _threshold_clause(query: str, match: re.Match[str]) -> str:
    """Return the local clause that gives a percentage its semantic subject."""
    preceding_boundaries = list(
        _THRESHOLD_CLAUSE_BOUNDARY_RE.finditer(query, 0, match.start())
    )
    start = preceding_boundaries[-1].end() if preceding_boundaries else 0
    following_boundary = _THRESHOLD_CLAUSE_BOUNDARY_RE.search(query, match.end())
    end = following_boundary.start() if following_boundary else len(query)
    return query[start:end].strip()


def _is_uncovered_threshold_clause(clause: str) -> bool:
    return bool(
        re.search(r"\b(?:share\s+of\s+import|import\s+share)\b", clause)
        or re.search(r"\b(?:uncovered|unknown-price\s+residual)\b", clause)
    )


def _is_covered_threshold_clause(clause: str) -> bool:
    explicitly_covered = bool(re.search(r"\b(?:covered|coverage)\b", clause))
    covered_basket = (
        "share" in clause
        and all(
            re.search(rf"\b{component}\b", clause)
            for component in ("ppa", "cfd", "regulated", "deregulated")
        )
    )
    return explicitly_covered or covered_basket


def extract_residual_coverage_threshold(query: str) -> Optional[ResidualCoverageThreshold]:
    """Parse the negligible-residual constraint from either framing.

    Returns ``None`` when no percentage constraint is scoped to import share,
    the uncovered residual, or the covered price basket. A returned threshold
    is NOT automatically negligible — callers gate on
    ``max_uncovered_share <= NEGLIGIBLE_UNCOVERED_SHARE_MAX`` so that
    "more than 5%" / "less than 20%" are parsed but correctly refused.
    """
    query_lower = str(query or "").strip().lower()
    if not query_lower:
        return None

    # Prefer a directly stated import/uncovered limit when both equivalent
    # framings appear. Percentages about price changes or other subjects are
    # ignored instead of becoming the residual threshold.
    uncovered = next(
        (
            match
            for match in _UNCOVERED_THRESHOLD_RE.finditer(query_lower)
            if _is_uncovered_threshold_clause(
                _threshold_clause(query_lower, match)
            )
        ),
        None,
    )
    if uncovered is not None:
        return ResidualCoverageThreshold(
            max_uncovered_share=round(float(uncovered.group("value")) / 100.0, 6),
            framing="uncovered",
            phrase=uncovered.group(0).strip(),
        )

    covered = next(
        (
            match
            for match in _COVERED_THRESHOLD_RE.finditer(query_lower)
            if _is_covered_threshold_clause(
                _threshold_clause(query_lower, match)
            )
        ),
        None,
    )
    if covered is not None:
        value = float(covered.group("value"))
        return ResidualCoverageThreshold(
            max_uncovered_share=round((100.0 - value) / 100.0, 6),
            framing="covered",
            phrase=covered.group(0).strip(),
        )
    return None


def resolve_import_share_filter(query: str) -> tuple[str, float, str]:
    """Return the month filter for the implied PPA/CfD approximation.

    Always expressed on the UNCOVERED (import) side — ``share_import <= X`` —
    whichever way the user phrased the constraint. Returning the raw
    coverage-side rule here would filter ``share_import > 0.995`` and match zero
    months. Falls back to the negligible bound when no explicit threshold is
    stated, matching the routing gate's own precondition.
    """
    threshold = extract_residual_coverage_threshold(query)
    if threshold is None:
        return "le", NEGLIGIBLE_UNCOVERED_SHARE_MAX, "was at most"
    return "le", threshold.max_uncovered_share, "was at most"


def is_implied_ppa_cfd_price_query(query: str) -> bool:
    """Recognize the bounded, deterministic negligible-residual approximation."""

    query_lower = str(query or "").strip().lower()
    if not query_lower:
        return False
    if not all(token in query_lower for token in ("balancing", "import", "ppa", "cfd")):
        return False
    if "share" not in query_lower:
        return False
    if not (re.search(r"\bregulated\b", query_lower) and re.search(r"\bderegulated\b", query_lower)):
        return False
    if not any(token in query_lower for token in _PRICE_INTENT_PHRASES):
        return False

    threshold = extract_residual_coverage_threshold(query_lower)
    if threshold is None:
        return False
    return threshold.max_uncovered_share <= NEGLIGIBLE_UNCOVERED_SHARE_MAX
