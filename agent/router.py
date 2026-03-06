"""
Phase 3 fast router: deterministic pre-LLM tool routing.
"""
import re
import os
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from agent.tools.types import ToolInvocation

ROUTER_ENABLE_SEMANTIC_FALLBACK = os.getenv("ROUTER_ENABLE_SEMANTIC_FALLBACK", "true").lower() in ("1", "true", "yes", "on")
ROUTER_SEMANTIC_MIN_SCORE = min(1.0, max(0.1, float(os.getenv("ROUTER_SEMANTIC_MIN_SCORE", "0.62"))))


ALLOWED_BALANCING_ENTITIES = {
    "import",
    "deregulated_hydro",
    "regulated_hpp",
    "regulated_new_tpp",
    "regulated_old_tpp",
    "renewable_ppa",
    "thermal_ppa",
}
ALLOWED_TARIFF_ENTITY_ALIASES = {"enguri", "gardabani_tpp", "old_tpp_group"}


MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_SEMANTIC_TOOL_TERMS: Dict[str, Set[str]] = {
    "get_prices": {
        "price", "prices", "cost", "costs", "balancing price", "deregulated",
        "gcap", "guaranteed capacity", "exchange rate", "xrate", "usd", "gel",
        "trend", "trends", "evolution", "evolve", "dynamics", "volatility",
        "annual", "annually", "balancing electricity",
    },
    "get_tariffs": {
        "tariff", "tariffs", "regulated", "gnerc", "enguri", "gardabani",
        "old tpp", "capacity fee", "cost-plus", "thermal plant tariff",
    },
    "get_generation_mix": {
        "generation", "generated", "output", "technology", "tech", "hydro",
        "thermal", "wind", "solar", "demand", "consumption", "losses",
        "supply", "import dependence",
    },
    "get_balancing_composition": {
        "share", "shares", "composition", "mix", "proportion", "weight",
        "contribution", "balancing electricity", "balancing market", "ppa",
        "import share", "entity share",
    },
}


def _tokenize_words(query_lower: str) -> Set[str]:
    return set(re.findall(r"[a-zA-Z_]+", query_lower))


def _semantic_score(query_lower: str, words: Set[str], terms: Set[str]) -> float:
    if not terms:
        return 0.0
    raw_hits = 0.0
    for term in terms:
        t = term.lower().strip()
        if not t:
            continue
        if " " in t:
            if t in query_lower:
                raw_hits += 2.0
        elif t in words:
            raw_hits += 1.0
    denom = max(2.0, len(terms) * 0.35)
    return min(1.0, raw_hits / denom)


def _build_semantic_invocation(
    tool_name: str,
    query_lower: str,
    start_date: Optional[str],
    end_date: Optional[str],
    score: float,
) -> Optional[ToolInvocation]:
    has_share = any(t in query_lower for t in ["share", "shares", "proportion", "percentage", "percent"])
    has_balancing = any(t in query_lower for t in ["balancing", "p_bal", "balance market", "balancing electricity"])
    if tool_name == "get_tariffs":
        entities = _extract_tariff_entities(query_lower)
        currency = _extract_currency(query_lower)
        if currency == "both":
            currency = "gel"
        return ToolInvocation(
            name="get_tariffs",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "entities": entities or None,
                "currency": currency,
            },
            confidence=min(0.95, max(0.65, score)),
            reason=f"Semantic fallback matched tariff intent (score={score:.2f})",
        )

    if tool_name == "get_balancing_composition":
        if not has_balancing:
            return None
        entities = _extract_balancing_entities(query_lower)
        return ToolInvocation(
            name="get_balancing_composition",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "entities": entities or None,
            },
            confidence=min(0.95, max(0.65, score)),
            reason=f"Semantic fallback matched balancing composition intent (score={score:.2f})",
        )

    if tool_name == "get_generation_mix":
        types = _extract_generation_types(query_lower)
        granularity = "yearly" if any(t in query_lower for t in ["yearly", "annual"]) else "monthly"
        return ToolInvocation(
            name="get_generation_mix",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "types": types or None,
                "mode": "share" if has_share else "quantity",
                "granularity": granularity,
            },
            confidence=min(0.95, max(0.65, score)),
            reason=f"Semantic fallback matched generation intent (score={score:.2f})",
        )

    if tool_name == "get_prices":
        metric = _extract_price_metric(query_lower)
        granularity = "yearly" if any(t in query_lower for t in ["yearly", "annual"]) else "monthly"
        return ToolInvocation(
            name="get_prices",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "currency": _extract_currency(query_lower),
                "metric": metric,
                "granularity": granularity,
            },
            confidence=min(0.95, max(0.65, score)),
            reason=f"Semantic fallback matched price intent (score={score:.2f})",
        )
    return None


def _semantic_match_tool(query_lower: str, start_date: Optional[str], end_date: Optional[str]) -> Optional[ToolInvocation]:
    words = _tokenize_words(query_lower)
    scores: List[Tuple[str, float]] = []
    for tool_name, terms in _SEMANTIC_TOOL_TERMS.items():
        score = _semantic_score(query_lower, words, terms)
        scores.append((tool_name, score))
    ranked = sorted(scores, key=lambda item: item[1], reverse=True)
    if not ranked:
        return None
    top_tool, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_score < ROUTER_SEMANTIC_MIN_SCORE:
        return None
    if (top_score - second_score) < 0.08:
        return None
    return _build_semantic_invocation(top_tool, query_lower, start_date, end_date, top_score)


def _extract_date_range(query_lower: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract coarse date range hints from NL query."""
    # from 2020 to 2024 / between 2020 and 2024
    m = re.search(r"(?:from|between)\s+(20\d{2})\s+(?:to|and|\-)\s+(20\d{2})", query_lower)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        start_year, end_year = min(y1, y2), max(y1, y2)
        return f"{start_year}-01-01", f"{end_year}-12-31"

    # range like 2020-2024
    m = re.search(r"\b(20\d{2})\s*-\s*(20\d{2})\b", query_lower)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        start_year, end_year = min(y1, y2), max(y1, y2)
        return f"{start_year}-01-01", f"{end_year}-12-31"

    # month + year, e.g. june 2024
    m = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(20\d{2})\b",
        query_lower,
    )
    if m:
        month = MONTH_MAP[m.group(1)]
        year = int(m.group(2))
        start = f"{year}-{month:02d}-01"
        return start, start

    # explicit year
    years = re.findall(r"\b(20\d{2})\b", query_lower)
    if years:
        year = int(years[0])
        return f"{year}-01-01", f"{year}-12-31"

    # relative "last N years"
    m = re.search(r"\blast\s+(\d+)\s+years?\b", query_lower)
    if m:
        n = max(1, int(m.group(1)))
        end_year = datetime.utcnow().year
        start_year = end_year - n + 1
        return f"{start_year}-01-01", f"{end_year}-12-31"

    return None, None


def _extract_currency(query_lower: str) -> str:
    has_usd = any(t in query_lower for t in ["usd", "dollar", "доллар", "დოლარ"])
    has_gel = any(t in query_lower for t in ["gel", "lari", "ლარი", "лари"])
    if has_usd and has_gel:
        return "both"
    if has_usd:
        return "usd"
    return "gel"


def _extract_price_metric(query_lower: str) -> str:
    if any(t in query_lower for t in ["dereg", "დერეგულ", "дерег"]):
        return "deregulated"
    if any(t in query_lower for t in ["gcap", "guaranteed capacity", "გარანტირებულ", "гарантирован"]):
        return "guaranteed_capacity"
    if any(t in query_lower for t in ["xrate", "exchange rate", "კურს", "курс"]):
        return "exchange_rate"
    return "balancing"


def _extract_balancing_entities(query_lower: str) -> List[str]:
    entity_map = {
        "import": "import",
        "deregulated hydro": "deregulated_hydro",
        "deregulated_hydro": "deregulated_hydro",
        "regulated hpp": "regulated_hpp",
        "regulated_hpp": "regulated_hpp",
        "new tpp": "regulated_new_tpp",
        "old tpp": "regulated_old_tpp",
        "renewable ppa": "renewable_ppa",
        "thermal ppa": "thermal_ppa",
        "renewable_ppa": "renewable_ppa",
        "thermal_ppa": "thermal_ppa",
    }
    hits = []
    for key, value in entity_map.items():
        if key in query_lower and value in ALLOWED_BALANCING_ENTITIES:
            hits.append(value)
    return list(dict.fromkeys(hits))


def _extract_tariff_entities(query_lower: str) -> List[str]:
    hits = []
    if any(t in query_lower for t in ["enguri", "ენგურ", "энгур"]):
        hits.append("enguri")
    if any(t in query_lower for t in ["gardabani", "გარდაბ", "гардаб"]):
        hits.append("gardabani_tpp")
    if any(t in query_lower for t in ["old tpp", "old thermal", "ძველი თეს", "стар"]):
        hits.append("old_tpp_group")
    return [h for h in hits if h in ALLOWED_TARIFF_ENTITY_ALIASES]


def _extract_generation_types(query_lower: str) -> List[str]:
    type_map = {
        "hydro": "hydro",
        "thermal": "thermal",
        "wind": "wind",
        "solar": "solar",
        "import": "import",
        "export": "export",
        "loss": "losses",
        "abkhaz": "abkhazeti",
        "direct customer": "direct customers",
        "distribution": "supply-distribution",
    }
    hits = []
    for key, value in type_map.items():
        if key in query_lower:
            hits.append(value)
    return list(dict.fromkeys(hits))


def match_tool(query: str) -> Optional[ToolInvocation]:
    """Return a deterministic tool invocation when confidence is high."""
    q = query.lower().strip()
    start_date, end_date = _extract_date_range(q)

    composition_terms = ["share", "composition", "mix", "proportion", "წილი", "доля"]
    explicit_share_terms = ["share", "proportion", "percentage", "percent", "წილი", "доля"]
    balancing_terms = ["balancing", "p_bal", "საბალანსო", "баланс"]
    tariff_terms = ["tariff", "ტარიფ", "тариф"]
    generation_terms = ["generation", "technology", "type_tech", "demand", "consumption", "გენერ", "потреб"]
    price_terms = ["price", "p_bal", "p_dereg", "p_gcap", "xrate", "exchange rate", "ფასი", "цена", "курс"]

    has_composition = any(t in q for t in composition_terms)
    has_share = any(t in q for t in explicit_share_terms)
    has_balancing = any(t in q for t in balancing_terms)
    has_tariff = any(t in q for t in tariff_terms)
    has_generation = any(t in q for t in generation_terms)
    has_price = any(t in q for t in price_terms)

    # 1) Tariffs
    if has_tariff or any(t in q for t in ["enguri", "gardabani", "ენგურ", "გარდაბ", "энгур", "гардаб"]):
        entities = _extract_tariff_entities(q)
        currency = _extract_currency(q)
        if currency == "both":
            currency = "gel"
        return ToolInvocation(
            name="get_tariffs",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "entities": entities or None,
                "currency": currency,
            },
            confidence=0.92 if has_tariff else 0.84,
            reason="Matched tariff-focused query",
        )

    # 2) Balancing composition
    if has_composition and has_balancing:
        entities = _extract_balancing_entities(q)
        return ToolInvocation(
            name="get_balancing_composition",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "entities": entities or None,
            },
            confidence=0.94,
            reason="Matched balancing composition query",
        )

    # 3) Generation mix / demand quantities
    if has_generation and not has_tariff:
        types = _extract_generation_types(q)
        mode = "share" if has_share else "quantity"
        granularity = "yearly" if any(t in q for t in ["yearly", "annual", "წლიურ", "год"]) else "monthly"
        return ToolInvocation(
            name="get_generation_mix",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "types": types or None,
                "mode": mode,
                "granularity": granularity,
            },
            confidence=0.85,
            reason="Matched generation/demand query",
        )

    # 4) Prices
    if has_price and not has_tariff:
        metric = _extract_price_metric(q)
        granularity = "yearly" if any(t in q for t in ["yearly", "annual", "წლიურ", "год"]) else "monthly"
        return ToolInvocation(
            name="get_prices",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "currency": _extract_currency(q),
                "metric": metric,
                "granularity": granularity,
            },
            confidence=0.82,
            reason="Matched price query",
        )

    if ROUTER_ENABLE_SEMANTIC_FALLBACK:
        semantic = _semantic_match_tool(q, start_date, end_date)
        if semantic:
            return semantic

    return None
