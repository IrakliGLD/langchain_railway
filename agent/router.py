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
    "CfD_scheme",
}
ALLOWED_TARIFF_ENTITY_ALIASES = {
    "enguri_hpp",
    "enguri",
    "vardnili_hpp",
    "vardnili",
    "dzevrula_hpp",
    "dzevruli_hpp",
    "gumati_hpp",
    "shaori_hpp",
    "rioni_hpp",
    "lajanuri_hpp",
    "zhinvali_hpp",
    "vartsikhe_hpp",
    "khramhesi_i",
    "khramhesi_ii",
    "gardabani_tpp",
    "mktvari_tpp",
    "mtkvari_tpp",
    "tbilsresi_tpp",
    "tbilresi_tpp",
    "gpower_tpp",
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


_MONTH_YEAR_PATTERN = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\s+(20\d{2})\b",
    re.IGNORECASE,
)

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
        "supply", "import dependence", "import dependency", "energy security",
        "self-sufficiency",
    },
    "get_balancing_composition": {
        "share", "shares", "composition", "mix", "proportion", "weight",
        "contribution", "balancing electricity", "balancing market", "ppa",
        "cfd", "hydro", "thermal", "deregulated", "regulated",
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
    query_text = str(query_lower or "")
    if not query_text:
        return None, None
    query_lower = query_text.lower()

    explicit_months: list[tuple[int, int]] = []
    seen_months: set[tuple[int, int]] = set()
    for match in _MONTH_YEAR_PATTERN.finditer(query_text):
        month_token = match.group(1).lower()
        year = int(match.group(2))
        month_num = MONTH_MAP.get(month_token)
        if not month_num:
            continue
        key = (year, month_num)
        if key in seen_months:
            continue
        seen_months.add(key)
        explicit_months.append(key)

    if explicit_months:
        start_year, start_month = min(explicit_months)
        end_year, end_month = max(explicit_months)
        end = f"{end_year}-{end_month:02d}-01"
        if is_explanation and len(explicit_months) == 1:
            # Shift back 1 year and 1 month to support MoM and YoY enrichment
            start_month_adj = start_month - 1
            start_year_adj = start_year - 5
            if start_month_adj == 0:
                start_month_adj = 12
                start_year_adj -= 1
            start = f"{start_year_adj}-{start_month_adj:02d}-01"
            return start, end
        return f"{start_year}-{start_month:02d}-01", end

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


_BALANCING_ENTITY_ALIAS_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("renewable ppa", ("renewable_ppa",)),
    ("renewable ppas", ("renewable_ppa",)),
    ("renewable_ppa", ("renewable_ppa",)),
    ("thermal generation ppa", ("thermal_ppa",)),
    ("thermal generation ppas", ("thermal_ppa",)),
    ("thermal ppa", ("thermal_ppa",)),
    ("thermal ppas", ("thermal_ppa",)),
    ("thermal_ppa", ("thermal_ppa",)),
    ("cfd scheme", ("CfD_scheme",)),
    ("cfd_scheme", ("CfD_scheme",)),
    ("support scheme cfd", ("CfD_scheme",)),
    ("contract for difference", ("CfD_scheme",)),
    ("imports", ("import",)),
    ("import", ("import",)),
    ("deregulated hydro generation", ("deregulated_hydro",)),
    ("deregulated hydropower", ("deregulated_hydro",)),
    ("deregulated hydro", ("deregulated_hydro",)),
    ("deregulated_hydro", ("deregulated_hydro",)),
    ("deregulated plants", ("deregulated_hydro",)),
    ("deregulated plant", ("deregulated_hydro",)),
    ("deregulated power plants", ("deregulated_hydro",)),
    ("deregulated power plant", ("deregulated_hydro",)),
    ("regulated hydro generation", ("regulated_hpp",)),
    ("regulated hydropower", ("regulated_hpp",)),
    ("regulated hydro", ("regulated_hpp",)),
    ("regulated hpps", ("regulated_hpp",)),
    ("regulated hpp", ("regulated_hpp",)),
    ("regulated_hpp", ("regulated_hpp",)),
    ("regulated new tpp", ("regulated_new_tpp",)),
    ("new regulated tpp", ("regulated_new_tpp",)),
    ("new tpp", ("regulated_new_tpp",)),
    ("regulated_new_tpp", ("regulated_new_tpp",)),
    ("regulated old tpp", ("regulated_old_tpp",)),
    ("old regulated tpp", ("regulated_old_tpp",)),
    ("old tpp", ("regulated_old_tpp",)),
    ("regulated_old_tpp", ("regulated_old_tpp",)),
    ("all regulated thermal", ("regulated_new_tpp", "regulated_old_tpp")),
    ("all regulated thermals", ("regulated_new_tpp", "regulated_old_tpp")),
    ("regulated thermal", ("regulated_new_tpp", "regulated_old_tpp")),
    ("regulated thermals", ("regulated_new_tpp", "regulated_old_tpp")),
    ("regulated tpp", ("regulated_new_tpp", "regulated_old_tpp")),
    ("regulated tpps", ("regulated_new_tpp", "regulated_old_tpp")),
    ("ppa cfd import residual", ("renewable_ppa", "thermal_ppa", "CfD_scheme", "import")),
    ("residual ppa/cfd/import", ("renewable_ppa", "thermal_ppa", "CfD_scheme", "import")),
    ("remaining electricity", ("renewable_ppa", "thermal_ppa", "CfD_scheme", "import")),
    ("remaining energy", ("renewable_ppa", "thermal_ppa", "CfD_scheme", "import")),
)


def extract_balancing_entities(query_lower: str) -> List[str]:
    positioned_hits: List[tuple[int, int, int, tuple[str, ...]]] = []
    for order, (key, values) in enumerate(_BALANCING_ENTITY_ALIAS_MAP):
        pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(key)}(?![a-z0-9_])")
        for match in pattern.finditer(query_lower):
            if any(value in ALLOWED_BALANCING_ENTITIES for value in values):
                positioned_hits.append((match.start(), -(match.end() - match.start()), order, values))

    hits: List[str] = []
    seen: Set[str] = set()
    occupied_spans: List[tuple[int, int]] = []
    for start, neg_len, _, values in sorted(positioned_hits):
        end = start - neg_len
        if any(not (end <= span_start or start >= span_end) for span_start, span_end in occupied_spans):
            continue
        occupied_spans.append((start, end))
        for value in values:
            if value not in ALLOWED_BALANCING_ENTITIES or value in seen:
                continue
            seen.add(value)
            hits.append(value)
    return hits


def extract_tariff_entities(query_lower: str) -> List[str]:
    hits = []
    if any(t in query_lower for t in ["enguri", "ენგურ", "энгур"]):
        hits.append("enguri")
    if "vardnili" in query_lower:
        hits.append("vardnili")
    if any(t in query_lower for t in ["dzevruli", "dzevrula", "dzevrul"]):
        hits.append("dzevruli_hpp")
    if "gumati" in query_lower:
        hits.append("gumati_hpp")
    if "shaori" in query_lower:
        hits.append("shaori_hpp")
    if "rioni" in query_lower:
        hits.append("rioni_hpp")
    if "lajanuri" in query_lower:
        hits.append("lajanuri_hpp")
    if any(t in query_lower for t in ["zhinvali", "zhinval"]):
        hits.append("zhinvali_hpp")
    if "vartsikhe" in query_lower:
        hits.append("vartsikhe_hpp")
    if any(t in query_lower for t in ["khramhesi i", "khrami i"]):
        hits.append("khramhesi_i")
    if any(t in query_lower for t in ["khramhesi ii", "khrami ii"]):
        hits.append("khramhesi_ii")
    if any(t in query_lower for t in ["gardabani", "გარდაბ", "гардаб"]):
        hits.append("gardabani_tpp")
    if any(t in query_lower for t in ["mktvari", "mtkvari"]):
        hits.append("mtkvari_tpp")
    if any(t in query_lower for t in ["tbilsresi", "tbilresi", "iec", "tbilisi tpp"]):
        hits.append("tbilresi_tpp")
    if any(t in query_lower for t in ["g-power", "g power", "gpower"]):
        hits.append("gpower_tpp")
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
