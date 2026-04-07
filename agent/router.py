"""
Phase 3 fast router: deterministic pre-LLM tool routing.
"""
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

_TZ_GEORGIA = ZoneInfo("Asia/Tbilisi")

from agent.tools.types import ToolInvocation
from config import ROUTER_ENABLE_SEMANTIC_FALLBACK, ROUTER_SEMANTIC_GAP_THRESHOLD, ROUTER_SEMANTIC_MIN_SCORE

# Populated by _semantic_match_tool on each call so pipeline.py can
# include scores in the miss-detail trace event.
_last_semantic_scores: Dict[str, float] = {}


# Public entity vocabularies keep deterministic extraction aligned with tool enums.
ALLOWED_BALANCING_ENTITIES = {
    "import",
    "deregulated_hydro",
    "regulated_hpp",
    "regulated_new_tpp",
    "regulated_old_tpp",
    "renewable_ppa",
    "thermal_ppa",
}
ALLOWED_TARIFF_ENTITY_ALIASES = {
    "enguri",
    "gardabani_tpp",
    "old_tpp_group",
    "regulated_hpp",
    "regulated_new_tpp",
    "regulated_old_tpp",
    "regulated_plants",
}


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


# Lightweight semantic scoring helps recover from keyword misses without using an LLM.
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
    # Use the same parameter extractors as the direct router so fallback behavior stays deterministic.
    has_share = any(t in query_lower for t in ["share", "shares", "proportion", "percentage", "percent"])
    has_balancing = any(t in query_lower for t in ["balancing", "p_bal", "balance market", "balancing electricity"])
    if tool_name == "get_tariffs":
        entities = extract_tariff_entities(query_lower)
        currency = extract_currency(query_lower)
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
        entities = extract_balancing_entities(query_lower)
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
        types = extract_generation_types(query_lower)
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
        metric = extract_price_metric(query_lower)
        granularity = "yearly" if any(t in query_lower for t in ["yearly", "annual"]) else "monthly"
        return ToolInvocation(
            name="get_prices",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "currency": extract_currency(query_lower),
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
    # Expose scores for observability (pipeline.py reads these on miss).
    _last_semantic_scores.clear()
    for tool_name, score in ranked:
        _last_semantic_scores[tool_name] = round(score, 3)
    if not ranked:
        return None
    top_tool, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_score < ROUTER_SEMANTIC_MIN_SCORE:
        return None
    if (top_score - second_score) < ROUTER_SEMANTIC_GAP_THRESHOLD:
        return None
    return _build_semantic_invocation(top_tool, query_lower, start_date, end_date, top_score)


# Regex extractors below convert natural-language hints into concrete tool params.
def extract_date_range(query_lower: str, is_explanation: bool = False) -> Tuple[Optional[str], Optional[str]]:
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

    # month-year range, e.g. jan 2023 to dec 2024
    m = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(20\d{2})\s+(?:to|and|\-\s|until)\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(20\d{2})\b",
        query_lower,
    )
    if m:
        m1 = MONTH_MAP[m.group(1)]
        y1 = int(m.group(2))
        m2 = MONTH_MAP[m.group(3)]
        y2 = int(m.group(4))
        # normalize order
        if (y1 > y2) or (y1 == y2 and m1 > m2):
            y1, m1, y2, m2 = y2, m2, y1, m1
        return f"{y1}-{m1:02d}-01", f"{y2}-{m2:02d}-01"

    # month + year, e.g. june 2024
    m = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(20\d{2})\b",
        query_lower,
    )
    if m:
        month = MONTH_MAP[m.group(1)]
        year = int(m.group(2))
        end = f"{year}-{month:02d}-01"
        if is_explanation:
            # Shift back 1 year and 1 month to support MoM and YoY enrichment
            start_month = month - 1
            start_year = year - 5
            if start_month == 0:
                start_month = 12
                start_year -= 1
            start = f"{start_year}-{start_month:02d}-01"
            return start, end
        return end, end

    # explicit year
    years = re.findall(r"\b(20\d{2})\b", query_lower)
    if years:
        year = int(years[0])
        end = f"{year}-12-31"
        if is_explanation:
            return f"{year-5}-01-01", end
        return f"{year}-01-01", end

    # relative "last N years"
    m = re.search(r"\blast\s+(\d+)\s+years?\b", query_lower)
    if m:
        n = max(1, int(m.group(1)))
        end_year = datetime.now(tz=_TZ_GEORGIA).year
        start_year = end_year - n + 1
        return f"{start_year}-01-01", f"{end_year}-12-31"

    # relative "last N months"
    m = re.search(r"\blast\s+(\d+)\s+months?\b", query_lower)
    if m:
        n = max(1, int(m.group(1)))
        now = datetime.now(tz=_TZ_GEORGIA)
        end_year, end_month = now.year, now.month
        # Walk back N months
        start_month = end_month - n
        start_year = end_year
        while start_month <= 0:
            start_month += 12
            start_year -= 1
        return f"{start_year}-{start_month:02d}-01", f"{end_year}-{end_month:02d}-01"

    return None, None


def extract_currency(query_lower: str) -> str:
    has_usd = any(t in query_lower for t in ["usd", "dollar", "доллар", "დოლარ"])
    has_gel = any(t in query_lower for t in ["gel", "lari", "ლარი", "лари"])
    if has_usd and has_gel:
        return "both"
    if has_usd:
        return "usd"
    return "gel"


def extract_price_metric(query_lower: str) -> str:
    if any(t in query_lower for t in ["dereg", "დერეგულ", "дерег"]):
        return "deregulated"
    if any(t in query_lower for t in ["gcap", "guaranteed capacity", "გარანტირებულ", "гарантирован"]):
        return "guaranteed_capacity"
    if any(t in query_lower for t in ["xrate", "exchange rate", "კურს", "курс"]):
        return "exchange_rate"
    return "balancing"


def extract_balancing_entities(query_lower: str) -> List[str]:
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


def extract_tariff_entities(query_lower: str) -> List[str]:
    hits = []
    if any(t in query_lower for t in ["enguri", "ენგურ", "энгур"]):
        hits.append("enguri")
    if any(t in query_lower for t in ["gardabani", "გარდაბ", "гардаб"]):
        hits.append("gardabani_tpp")
    if any(t in query_lower for t in ["regulated hpp", "regulated_hpp"]):
        hits.append("regulated_hpp")
    if any(t in query_lower for t in ["new tpp", "regulated_new_tpp"]):
        hits.append("regulated_new_tpp")
    if any(t in query_lower for t in ["old tpp", "old thermal", "ძველი თეს", "стар"]):
        hits.append("old_tpp_group")
        hits.append("regulated_old_tpp")
    # Broad regulated-plant lookups should fetch all regulated tariff buckets
    # instead of falling back to the narrow Enguri/Gardabani defaults.
    generic_regulated_lookup = (
        any(
            token in query_lower
            for token in (
                "under regulation",
                "under price regulation",
                "price regulation",
                "regulated plant",
                "regulated plants",
                "regulated power plant",
                "regulated power plants",
            )
        )
        or (
            "regulated" in query_lower
            and any(token in query_lower for token in ("plant", "plants", "entity", "entities"))
        )
    )
    list_shaped_lookup = any(
        token in query_lower
        for token in ("which", "what are the", "list", "show all", "enumerate", "name")
    )
    mentions_specific_alias = any(
        token in query_lower
        for token in (
            "enguri",
            "gardabani",
            "regulated hpp",
            "regulated_hpp",
            "new tpp",
            "regulated_new_tpp",
            "old tpp",
            "old thermal",
        )
    )
    if generic_regulated_lookup and not mentions_specific_alias:
        hits.extend(["regulated_hpp", "regulated_new_tpp", "regulated_old_tpp"])

    return [h for h in dict.fromkeys(hits) if h in ALLOWED_TARIFF_ENTITY_ALIASES]


def extract_generation_types(query_lower: str) -> List[str]:
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


# Main deterministic routing ladder: tariffs, composition, generation, prices, then semantic fallback.
def match_tool(query: str, is_explanation: bool = False) -> Optional[ToolInvocation]:
    """Return a deterministic tool invocation when confidence is high."""
    q = query.lower().strip()
    start_date, end_date = extract_date_range(q, is_explanation=is_explanation)

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
        entities = extract_tariff_entities(q)
        currency = extract_currency(q)
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
        entities = extract_balancing_entities(q)
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
        types = extract_generation_types(q)
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
        metric = extract_price_metric(q)
        granularity = "yearly" if any(t in q for t in ["yearly", "annual", "წლიურ", "год"]) else "monthly"
        return ToolInvocation(
            name="get_prices",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "currency": extract_currency(q),
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
