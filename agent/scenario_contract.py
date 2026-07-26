"""Deterministic grounding for numerical scenario requests.

The question analyzer may propose a scenario contract, but this module is the
authority for whether the numerical parameter and transformed subject actually
occur in the user's question. It intentionally fails closed: ambiguous subjects,
date-only numbers, currency mismatches, and ungrounded factors produce no
scenario request.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping

from config_metrics.metric_units import metric_is_additive


_SCENARIO_METRICS = frozenset({
    "scenario_scale",
    "scenario_offset",
    "scenario_payoff",
})

_SUBJECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("guaranteed_capacity", re.compile(r"\bguaranteed\s+capacity(?:\s+price)?\b", re.I)),
    ("deregulated", re.compile(r"\bderegulated(?:\s+market)?(?:\s+price)?\b", re.I)),
    ("balancing", re.compile(r"\bbalancing(?:\s+electricity)?(?:\s+price)?\b", re.I)),
    ("exchange_rate", re.compile(r"\b(?:exchange|fx)\s+rate\b|\bgel\s*/\s*usd\b", re.I)),
    ("ppa", re.compile(r"\bppa(?:s)?(?:\s+share)?\b|\bpower\s+purchase\s+agreements?\b", re.I)),
    ("cfd", re.compile(r"\bcfd(?:s)?(?:\s+share)?\b|\bcontracts?\s+for\s+difference\b", re.I)),
    ("import", re.compile(r"\bimports?(?:\s+share)?\b|\bimport\s+share\b", re.I)),
    ("thermal", re.compile(r"\bthermal(?:\s+(?:generation|output|share))?\b", re.I)),
    ("renewable", re.compile(r"\brenewables?(?:\s+(?:generation|output|share))?\b", re.I)),
    ("generation", re.compile(r"\b(?:generation|output)\b", re.I)),
    ("demand", re.compile(r"\b(?:demand|consumption)\b", re.I)),
    ("generic_price", re.compile(r"\bprices?\b", re.I)),
)

_METRIC_FAMILIES: dict[str, str] = {
    "balancing": "balancing",
    "p_bal_gel": "balancing",
    "p_bal_usd": "balancing",
    "deregulated": "deregulated",
    "p_dereg_gel": "deregulated",
    "p_dereg_usd": "deregulated",
    "guaranteed_capacity": "guaranteed_capacity",
    "p_gcap_gel": "guaranteed_capacity",
    "p_gcap_usd": "guaranteed_capacity",
    "exchange_rate": "exchange_rate",
    "xrate": "exchange_rate",
    "import": "import",
    "share_import": "import",
    "ppa": "ppa",
    "share_all_ppa": "ppa",
    "share_renewable_ppa": "ppa",
    "share_thermal_ppa": "ppa",
    "cfd": "cfd",
    "share_cfd_scheme": "cfd",
    "thermal": "thermal",
    "share_regulated_old_tpp": "thermal",
    "share_regulated_new_tpp": "thermal",
    "renewable": "renewable",
    "share_all_renewables": "renewable",
    "generation": "generation",
    "total_domestic_generation": "generation",
    "local_generation": "generation",
    "demand": "demand",
    "consumption": "demand",
    "total_demand": "demand",
}

_FALLBACK_METRIC_BY_FAMILY: dict[str, str] = {
    "balancing": "balancing",
    "deregulated": "deregulated",
    "guaranteed_capacity": "guaranteed_capacity",
    "exchange_rate": "exchange_rate",
    "import": "share_import",
    "ppa": "share_all_ppa",
    "cfd": "share_cfd_scheme",
    "thermal": "thermal",
    "renewable": "renewable",
    "generation": "generation",
    "demand": "demand",
}

_SETTLEMENT_PRICE_FAMILIES = frozenset({
    "balancing",
    "deregulated",
    "guaranteed_capacity",
})

_PERCENT_RE = re.compile(r"(?P<value>\d+(?:[.,]\d+)?)\s*(?:%|percent\b)", re.I)
_MULTIPLIER_RE = re.compile(
    r"\b(?P<word>double|doubles|doubling|twice|triple|triples|tripling|"
    r"quadruple|halve|halves|halving|half)\b",
    re.I,
)
_PRICE_AMOUNT_RE = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<currency>usd|gel|eur)"
    r"(?:\s*/\s*mwh)?",
    re.I,
)
_ENERGY_RE = re.compile(
    r"(?<![/\w])(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>kwh|mwh|gwh)\b",
    re.I,
)
_CAPACITY_RE = re.compile(
    r"(?<![/\w])(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>kw|mw|gw)\b",
    re.I,
)

_UP_WORDS = re.compile(
    r"\b(?:higher|more|increase[sd]?|increasing|rise[sd]?|rising|"
    r"raise[sd]?|raising|grow[sd]?|growing|up)\b",
    re.I,
)
_DOWN_WORDS = re.compile(
    r"\b(?:lower|less|decrease[sd]?|decreasing|fall[sd]?|falling|"
    r"drop[sd]?|dropping|reduce[sd]?|reducing|cut|cuts|down)\b",
    re.I,
)

_FULL_SERIES_RE = re.compile(
    r"\b(?:all|entire|full)\s+(?:history|series|dataset|period)\b", re.I
)
_PERIOD_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:from|between|during)\b|"
    r"\bover\s+(?:the\s+)?(?:period|months?|years?|quarters?|weeks?|days?)\b|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b",
    re.I,
)
_CROSS_METRIC_EFFECT_RE = re.compile(
    r"\b(?:affect|effect|impact|happen\s+to|what\s+will\s+happen|"
    r"how\s+will|change\s+the)\b",
    re.I,
)


def _number(value: str) -> float:
    return float(value.replace(",", "."))


def _direction(text: str, start: int, end: int) -> int | None:
    before = text[max(0, start - 55):start]
    after = text[end:min(len(text), end + 35)]
    neighborhood = f"{before} {after}"
    up = bool(_UP_WORDS.search(neighborhood))
    down = bool(_DOWN_WORDS.search(neighborhood))
    if up == down:
        return None
    return 1 if up else -1


def _nearest_subject(text: str, start: int, end: int) -> str | None:
    candidates: list[tuple[int, str]] = []
    for family, pattern in _SUBJECT_PATTERNS:
        for match in pattern.finditer(text):
            distance = min(abs(match.end() - start), abs(match.start() - end))
            if distance <= 90:
                candidates.append((distance, family))
    if not candidates:
        return None
    specific = [item for item in candidates if item[1] != "generic_price"]
    if specific:
        candidates = specific
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _metric_family(metric: object) -> str | None:
    normalized = str(metric or "").strip().lower()
    return _METRIC_FAMILIES.get(normalized)


def _payoff_reference_family(query: str) -> str | None:
    """Resolve the observed price series named as the payoff reference."""

    matches: list[tuple[int, str]] = []
    for family, pattern in _SUBJECT_PATTERNS:
        if family in {
            "ppa",
            "cfd",
            "import",
            "thermal",
            "renewable",
            "generation",
            "demand",
            "generic_price",
        }:
            continue
        match = pattern.search(query)
        if match:
            matches.append((match.start(), family))
    families = {family for _start, family in matches}
    if len(families) != 1:
        return None
    return matches[0][1]


def _fallback_metric(family: str, query: str) -> str | None:
    metric = _FALLBACK_METRIC_BY_FAMILY.get(family)
    if metric is None:
        return None
    query_lower = query.lower()
    if family in {"balancing", "deregulated", "guaranteed_capacity"}:
        suffix = ""
        if "usd" in query_lower and "gel" not in query_lower:
            suffix = "_usd"
        elif "gel" in query_lower and "usd" not in query_lower:
            suffix = "_gel"
        if suffix:
            prefixes = {
                "balancing": "p_bal",
                "deregulated": "p_dereg",
                "guaranteed_capacity": "p_gcap",
            }
            return f"{prefixes[family]}{suffix}"
    return metric


def _currency_for_metric(metric: object) -> str | None:
    normalized = str(metric or "").strip().lower()
    if normalized.endswith("_usd"):
        return "usd"
    if normalized.endswith("_gel"):
        return "gel"
    return None


def _scope_from_query(query: str) -> str:
    if _FULL_SERIES_RE.search(query):
        return "full_series"
    if _PERIOD_RE.search(query):
        return "requested_period"
    return "latest"


def _aggregation_from_query(
    query: str,
    *,
    metric_name: str,
    metric: str,
    has_energy: bool,
) -> str:
    if re.search(r"\b(?:average|mean)\b", query, re.I):
        return "mean"
    if re.search(r"\b(?:minimum|min|lowest)\b", query, re.I):
        return "min"
    if re.search(r"\b(?:maximum|max|highest)\b", query, re.I):
        return "max"
    if re.search(r"\b(?:sum|total|overall|combined)\b", query, re.I):
        if metric_name == "scenario_payoff":
            return "sum" if has_energy else "mean"
        return "sum" if metric_is_additive(metric) else "mean"
    if metric_name == "scenario_payoff" and has_energy:
        return "sum"
    return "mean"


def _energy_mwh(query: str) -> float | None:
    match = _ENERGY_RE.search(query)
    if not match:
        return None
    value = _number(match.group("value"))
    unit = match.group("unit").lower()
    if unit == "gwh":
        return value * 1000.0
    if unit == "kwh":
        return value / 1000.0
    return value


def _capacity_mw(query: str) -> float | None:
    match = _CAPACITY_RE.search(query)
    if not match:
        return None
    value = _number(match.group("value"))
    unit = match.group("unit").lower()
    if unit == "gw":
        return value * 1000.0
    if unit == "kw":
        return value / 1000.0
    return value


def is_cross_metric_counterfactual(query: str) -> bool:
    """Whether the question asks one changed metric to affect another metric."""

    text = str(query or "")
    if re.search(r"\b(?:payoff|compensation|strike)\b", text, re.I):
        return False
    families = {
        family
        for family, pattern in _SUBJECT_PATTERNS
        if family != "generic_price" and pattern.search(text)
    }
    return len(families) >= 2 and bool(_CROSS_METRIC_EFFECT_RE.search(text))


def _factor_candidates(query: str, metric_name: str) -> list[tuple[float, int, int, str | None]]:
    candidates: list[tuple[float, int, int, str | None]] = []
    if metric_name == "scenario_scale":
        for match in _PERCENT_RE.finditer(query):
            direction = _direction(query, match.start(), match.end())
            if direction is None:
                continue
            pct = _number(match.group("value"))
            factor = 1.0 + (direction * pct / 100.0)
            candidates.append((
                factor,
                match.start(),
                match.end(),
                _nearest_subject(query, match.start(), match.end()),
            ))
        multipliers = {
            "double": 2.0,
            "doubles": 2.0,
            "doubling": 2.0,
            "twice": 2.0,
            "triple": 3.0,
            "triples": 3.0,
            "tripling": 3.0,
            "quadruple": 4.0,
            "halve": 0.5,
            "halves": 0.5,
            "halving": 0.5,
            "half": 0.5,
        }
        for match in _MULTIPLIER_RE.finditer(query):
            candidates.append((
                multipliers[match.group("word").lower()],
                match.start(),
                match.end(),
                _nearest_subject(query, match.start(), match.end()),
            ))
    elif metric_name == "scenario_offset":
        for match in _PRICE_AMOUNT_RE.finditer(query):
            direction = _direction(query, match.start(), match.end())
            if direction is None:
                continue
            candidates.append((
                direction * _number(match.group("value")),
                match.start(),
                match.end(),
                _nearest_subject(query, match.start(), match.end()),
            ))
    elif metric_name == "scenario_payoff":
        if not re.search(
            r"\b(?:cfd|ppa|payoff|compensation|strike|contract\s+for\s+difference)\b",
            query,
            re.I,
        ):
            return []
        reference_family = _payoff_reference_family(query)
        if reference_family is None:
            return []
        for match in _PRICE_AMOUNT_RE.finditer(query):
            neighborhood = query[
                max(0, match.start() - 65):min(len(query), match.end() + 65)
            ]
            explicit_strike = bool(
                re.search(r"\bstrike(?:\s+price)?\b", neighborhood, re.I)
            )
            explicit_assumption = bool(
                re.search(
                    r"\b(?:at|with|for|assuming|assume)\b",
                    query[max(0, match.start() - 30):match.start()],
                    re.I,
                )
                and re.search(
                    r"\b(?:payoff|compensation|income|revenue)\b",
                    query,
                    re.I,
                )
            )
            if not explicit_strike and not explicit_assumption:
                continue
            candidates.append((
                _number(match.group("value")),
                match.start(),
                match.end(),
                reference_family,
            ))
    return candidates


def ground_scenario_request(
    raw_query: str,
    request: Mapping[str, Any],
    *,
    canonical_query: str | None = None,
) -> dict[str, Any] | None:
    """Return a query-grounded request or ``None`` when validation fails.

    ``canonical_query`` can help identify the subject for translated questions,
    but the raw query remains the numerical source of truth. At present the
    deterministic parser only accepts parameters it can read directly from
    ``raw_query``; it never trusts a number introduced by translation or by the
    question analyzer.
    """

    query = str(raw_query or "").strip()
    if not query:
        return None
    metric_name = str(request.get("metric_name") or "")
    if metric_name not in _SCENARIO_METRICS:
        return None
    if request.get("scenario_volume") is not None:
        return None
    if metric_name != "scenario_payoff" and (
        is_cross_metric_counterfactual(query)
        or (
            canonical_query is not None
            and is_cross_metric_counterfactual(canonical_query)
        )
    ):
        return None
    try:
        proposed_factor = float(request.get("scenario_factor"))
    except (TypeError, ValueError):
        return None
    if metric_name == "scenario_scale" and proposed_factor < 0:
        return None

    request_family = _metric_family(request.get("metric"))
    if request_family is None:
        return None

    candidates = _factor_candidates(query, metric_name)
    if not candidates and canonical_query and canonical_query != raw_query:
        # Translation may supply directional/subject vocabulary, but a value
        # candidate is accepted only when its numeric token is also present in
        # the raw question.
        raw_numbers = {_number(m.group(0)) for m in re.finditer(r"\d+(?:[.,]\d+)?", query)}
        candidates = [
            candidate
            for candidate in _factor_candidates(canonical_query, metric_name)
            if any(
                math.isclose(abs(candidate[0]), number, rel_tol=0.0, abs_tol=1e-9)
                or math.isclose(abs((candidate[0] - 1.0) * 100.0), number, rel_tol=0.0, abs_tol=1e-9)
                for number in raw_numbers
            )
        ]

    grounded: tuple[float, int, int, str | None] | None = None
    for candidate in candidates:
        factor, _start, _end, subject_family = candidate
        if subject_family != request_family:
            continue
        if math.isclose(factor, proposed_factor, rel_tol=0.0, abs_tol=1e-6):
            grounded = candidate
            break
    if grounded is None:
        return None

    result = dict(request)
    result["scenario_factor"] = grounded[0]
    result["scenario_scope"] = _scope_from_query(query)

    energy = _energy_mwh(query) if metric_name == "scenario_payoff" else None
    capacity = _capacity_mw(query) if metric_name == "scenario_payoff" else None
    result["scenario_energy_mwh"] = energy
    result["scenario_capacity_mw"] = capacity
    result["scenario_aggregation"] = _aggregation_from_query(
        query,
        metric_name=metric_name,
        metric=str(result.get("metric") or ""),
        has_energy=energy is not None,
    )

    if metric_name == "scenario_payoff" and request_family not in _SETTLEMENT_PRICE_FAMILIES:
        return None

    if (
        metric_name in {"scenario_offset", "scenario_payoff"}
        and request_family in _SETTLEMENT_PRICE_FAMILIES
    ):
        price_match = next(
            (
                match
                for match in _PRICE_AMOUNT_RE.finditer(query)
                if match.start() == grounded[1] and match.end() == grounded[2]
            ),
            None,
        )
        if price_match is None:
            return None
        requested_currency = _currency_for_metric(request.get("metric"))
        query_currency = price_match.group("currency").lower()
        if query_currency not in {"usd", "gel"}:
            return None
        if requested_currency is not None and requested_currency != query_currency:
            return None
        if requested_currency is None:
            normalized_metric = _fallback_metric(request_family, query)
            if normalized_metric is None or _currency_for_metric(normalized_metric) != query_currency:
                return None
            result["metric"] = normalized_metric

    return result


def ground_scenario_requests(
    raw_query: str,
    requests: Iterable[Mapping[str, Any]],
    *,
    canonical_query: str | None = None,
) -> list[dict[str, Any]]:
    """Ground all valid scenario requests while preserving non-scenario ones."""

    grounded: list[dict[str, Any]] = []
    for request in requests:
        metric_name = str(request.get("metric_name") or "")
        if metric_name not in _SCENARIO_METRICS:
            grounded.append(dict(request))
            continue
        normalized = ground_scenario_request(
            raw_query,
            request,
            canonical_query=canonical_query,
        )
        if normalized is not None:
            grounded.append(normalized)
    return grounded


def extract_scenario_requests(
    raw_query: str,
    *,
    canonical_query: str | None = None,
) -> list[dict[str, Any]]:
    """Extract one safe fallback request without inventing a target metric."""

    query = str(raw_query or "").strip()
    if not query:
        return []
    if is_cross_metric_counterfactual(query) or (
        canonical_query is not None
        and is_cross_metric_counterfactual(canonical_query)
    ):
        return []
    for metric_name in ("scenario_payoff", "scenario_scale", "scenario_offset"):
        for factor, _start, _end, family in _factor_candidates(query, metric_name):
            if family is None or family == "generic_price":
                continue
            metric = _fallback_metric(family, query)
            if metric is None:
                continue
            proposed = {
                "metric_name": metric_name,
                "metric": metric,
                "scenario_factor": factor,
            }
            grounded = ground_scenario_request(
                query,
                proposed,
                canonical_query=canonical_query,
            )
            if grounded is not None:
                return [grounded]
    return []


def query_has_scenario_parameter(query: str) -> bool:
    """Whether a query contains a directional parameter tied to a subject.

    This compatibility predicate is intentionally weaker than
    :func:`ground_scenario_request` because no proposed metric is available to
    compare against. It still excludes bare dates and free-floating quantities.
    """

    text = str(query or "").strip()
    if not text:
        return False
    return any(
        subject is not None
        for metric_name in _SCENARIO_METRICS
        for _factor, _start, _end, subject in _factor_candidates(text, metric_name)
    )
