"""
Knowledge module: Markdown-based domain knowledge with topic registry.

Replaces the monolithic domain_knowledge.py dict with individual .md files
and a simple keyword-to-file mapping for context selection.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

log = logging.getLogger("Enai")

# In-memory cache of loaded knowledge files
_KNOWLEDGE: Dict[str, str] = {}

# Full knowledge JSON cache (for backward compatibility with use_cache=True)
_KNOWLEDGE_JSON: str = ""


def load_knowledge() -> None:
    """Load all .md files at startup into a dict keyed by stem."""
    global _KNOWLEDGE, _KNOWLEDGE_JSON
    knowledge_dir = Path(__file__).parent
    count = 0
    for path in sorted(knowledge_dir.glob("*.md")):
        _KNOWLEDGE[path.stem] = path.read_text(encoding="utf-8")
        count += 1
    log.info(f"✅ Loaded {count} knowledge files from {knowledge_dir}")

    # Build backward-compatible JSON representation
    _rebuild_json_cache()


def _rebuild_json_cache() -> None:
    """Build a JSON string from all loaded knowledge for backward compat."""
    global _KNOWLEDGE_JSON
    knowledge_dict = {}
    for stem, content in _KNOWLEDGE.items():
        knowledge_dict[stem] = content
    _KNOWLEDGE_JSON = json.dumps(knowledge_dict, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Topic-to-file mapping
# ---------------------------------------------------------------------------
# Each keyword maps to a list of knowledge file stems that are relevant.
# This replaces the keyword-trigger dict in core/llm.py.

TOPIC_MAP: Dict[str, List[str]] = {
    # General definitions / conceptual questions
    "what is": ["general_definitions"],
    "what are": ["general_definitions"],
    "რა არის": ["general_definitions"],
    "что такое": ["general_definitions"],
    "define": ["general_definitions"],
    "explain": ["general_definitions"],
    "meaning of": ["general_definitions"],
    "განმარტე": ["general_definitions"],
    "объясни": ["general_definitions"],
    "renewable energy": ["general_definitions"],
    "განახლებადი ენერგია": ["general_definitions"],
    "electricity market": ["general_definitions"],
    "ელექტროენერგიის ბაზარი": ["general_definitions"],

    # Balancing price / market
    "balancing": ["balancing_price", "market_structure"],
    "p_bal": ["balancing_price"],
    "საბალანსო": ["balancing_price", "market_structure"],
    "баланс": ["balancing_price", "market_structure"],
    "price": ["balancing_price", "currency_influence"],
    "driver": ["balancing_price"],
    "composition": ["balancing_price", "market_structure"],
    "weighted": ["balancing_price"],
    "imbalance": ["market_structure"],
    "settlement": ["market_structure"],
    "decomposition": ["balancing_price"],
    "contribution": ["balancing_price"],

    # Tariffs
    "tariff": ["tariffs"],
    "ტარიფი": ["tariffs"],
    "тариф": ["tariffs"],
    "regulated": ["tariffs", "general_definitions"],
    "enguri": ["tariffs", "pso_trading"],
    "vardnili": ["tariffs", "pso_trading"],
    "gardabani": ["tariffs"],
    "gnerc": ["tariffs", "market_structure"],
    "cost-plus": ["tariffs"],
    "capacity fee": ["tariffs"],
    "engurhesi": ["tariffs"],

    # PSO Trading
    "pso": ["pso_trading", "tariffs"],
    "public service obligation": ["pso_trading"],
    "telmico": ["pso_trading"],
    "ep georgia": ["pso_trading"],
    "procurement": ["pso_trading"],
    "universal service": ["pso_trading"],
    "cascade distribution": ["pso_trading"],

    # CfD / PPA
    "cfd": ["cfd_ppa"],
    "contract for difference": ["cfd_ppa"],
    "strike price": ["cfd_ppa"],
    "ppa": ["cfd_ppa"],
    "power purchase agreement": ["cfd_ppa"],
    "support scheme": ["cfd_ppa"],
    "წახალისების სქემა": ["cfd_ppa"],

    # Currency / exchange rate
    "exchange rate": ["currency_influence"],
    "xrate": ["currency_influence", "balancing_price"],
    "gel": ["currency_influence"],
    "usd": ["currency_influence", "balancing_price"],
    "depreciation": ["currency_influence"],
    "გაცვლითი კურსი": ["currency_influence"],
    "обменный курс": ["currency_influence"],

    # Seasonal
    "season": ["seasonal_patterns"],
    "summer": ["seasonal_patterns", "balancing_price"],
    "winter": ["seasonal_patterns", "balancing_price"],
    "ზაფხულ": ["seasonal_patterns"],
    "ზამთარ": ["seasonal_patterns"],
    "сезон": ["seasonal_patterns"],

    # Generation
    "generation": ["generation_mix"],
    "გენერაცია": ["generation_mix"],
    "генерация": ["generation_mix"],
    "hydro": ["generation_mix", "balancing_price"],
    "thermal": ["generation_mix", "balancing_price"],
    "wind": ["generation_mix"],
    "solar": ["generation_mix"],
    "generation mix": ["generation_mix"],

    # Trade / import / export
    "import": ["market_structure", "currency_influence"],
    "export": ["market_structure"],
    "trade": ["market_structure"],
    "იმპორტი": ["market_structure"],
    "ექსპორტი": ["market_structure"],
    "импорт": ["market_structure"],
    "экспорт": ["market_structure"],
    "interconnection": ["market_structure"],
    "cross-border": ["market_structure"],

    # Market participants
    "esco": ["market_structure"],
    "gse": ["market_structure"],
    "genex": ["market_structure"],
    "geostat": ["market_structure"],
    "participant": ["market_structure"],

    # Forecasting
    "forecast": ["seasonal_patterns", "balancing_price", "sql_examples"],
    "predict": ["seasonal_patterns", "sql_examples"],
    "projection": ["seasonal_patterns", "sql_examples"],
    "trendline": ["seasonal_patterns", "sql_examples"],
    "პროგნოზი": ["seasonal_patterns", "sql_examples"],
    "прогноз": ["seasonal_patterns", "sql_examples"],

    # Energy security
    "energy security": ["generation_mix", "sql_examples"],
    "import dependence": ["generation_mix", "market_structure"],
    "self-sufficiency": ["generation_mix", "sql_examples"],

    # Abkhazeti
    "abkhaz": ["market_structure"],
    "აფხაზეთ": ["market_structure"],

    # Direct customers
    "direct customer": ["market_structure"],
    "პირდაპირი მომხმარებელი": ["market_structure"],
    "wholesale market": ["market_structure"],
    "metallurg": ["market_structure"],

    # CPI
    "cpi": ["currency_influence"],
    "inflation": ["currency_influence"],
    "ინფლაცია": ["currency_influence"],

    # Table selection (technical)
    "tech_quantity": ["generation_mix"],
    "trade_derived": ["market_structure"],
    "energy_balance": ["generation_mix"],
}


def get_knowledge_for_query(user_query: str) -> str:
    """Return concatenated knowledge for the topics matching the query.

    This is the primary interface for the LLM prompt builder.
    It replaces the keyword-trigger filtering in core/llm.py.

    Args:
        user_query: The user's natural language query.

    Returns:
        Concatenated Markdown content from matching knowledge files.
    """
    query_lower = user_query.lower()
    matched_files: set = set()

    for keyword, file_stems in TOPIC_MAP.items():
        if keyword in query_lower:
            matched_files.update(file_stems)

    # Fallback: if nothing matched, include balancing_price (most common)
    # and general_definitions (for conceptual questions)
    if not matched_files:
        definition_patterns = ["what is", "what are", "რა არის", "что такое",
                               "define", "explain"]
        is_conceptual = any(p in query_lower for p in definition_patterns)
        if is_conceptual:
            matched_files = {"general_definitions"}
        else:
            matched_files = {"balancing_price", "sql_examples"}

    # Build the output
    sections = []
    for stem in sorted(matched_files):
        content = _KNOWLEDGE.get(stem)
        if content:
            sections.append(content)
        else:
            log.warning(f"⚠️ Knowledge file '{stem}' not found in loaded knowledge")

    result = "\n\n---\n\n".join(sections)
    log.info(f"📚 Knowledge: matched {len(matched_files)} files "
             f"({', '.join(sorted(matched_files))}) for query")
    return result


def get_knowledge_json(user_query: str = "", use_cache: bool = True) -> str:
    """Backward-compatible interface matching old get_relevant_domain_knowledge().

    Args:
        user_query: The user's query text.
        use_cache: If True, return full cached JSON. If False, return filtered.

    Returns:
        JSON string of knowledge (full or filtered).
    """
    if use_cache:
        return _KNOWLEDGE_JSON

    # For filtered mode, convert matched Markdown to a JSON-serializable dict
    query_lower = user_query.lower()
    matched_files: set = set()

    for keyword, file_stems in TOPIC_MAP.items():
        if keyword in query_lower:
            matched_files.update(file_stems)

    if not matched_files:
        definition_patterns = ["what is", "what are", "რა არის", "что такое",
                               "define", "explain"]
        is_conceptual = any(p in query_lower for p in definition_patterns)
        if is_conceptual:
            matched_files = {"general_definitions"}
        else:
            matched_files = {"balancing_price", "sql_examples"}

    relevant = {}
    for stem in sorted(matched_files):
        content = _KNOWLEDGE.get(stem)
        if content:
            relevant[stem] = content

    return json.dumps(relevant, indent=2, ensure_ascii=False)


def get_knowledge_json_with_topics(
    preferred_topics: Optional[Iterable[str]],
    *,
    fallback_query: str = "",
    use_cache: bool = False,
) -> str:
    """Return knowledge JSON using preferred topics first, with query fallback."""

    if use_cache and not preferred_topics:
        return _KNOWLEDGE_JSON

    preferred = {
        topic_name
        for topic_name in (str(topic).strip() for topic in (preferred_topics or []))
        if topic_name in _KNOWLEDGE
    }

    if preferred:
        relevant = {
            stem: _KNOWLEDGE[stem]
            for stem in sorted(preferred)
            if stem in _KNOWLEDGE
        }
        return json.dumps(relevant, indent=2, ensure_ascii=False)

    return get_knowledge_json(fallback_query, use_cache=use_cache)


def get_knowledge_for_topics(
    preferred_topics: Optional[Iterable[str]],
    *,
    fallback_query: str = "",
) -> str:
    """Return concatenated Markdown using preferred topics first, with query fallback."""

    preferred = {
        topic_name
        for topic_name in (str(topic).strip() for topic in (preferred_topics or []))
        if topic_name in _KNOWLEDGE
    }
    if preferred:
        sections = [_KNOWLEDGE[stem] for stem in sorted(preferred) if stem in _KNOWLEDGE]
        return "\n\n---\n\n".join(sections)

    return get_knowledge_for_query(fallback_query)
