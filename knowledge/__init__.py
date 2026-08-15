"""
Knowledge module: Markdown-based domain knowledge with topic registry.

Replaces the monolithic domain_knowledge.py dict with individual .md files
and a simple keyword-to-file mapping for context selection.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
    "imbalance": ["market_structure", "exchange_transition"],
    "settlement": ["market_structure", "exchange_transition"],
    "decomposition": ["balancing_price"],
    "contribution": ["balancing_price"],

    # Tariffs
    # NOTE: bare "tariff" fans out to BOTH tariff files. tariffs.md is the
    # generation side (what a plant is paid, GEL/MWh); network_supply_tariffs.md
    # is the retail side (what a consumer pays, GEL/kWh). Routing "tariff" to
    # only one of them silently hides half the subject.
    "tariff": ["tariffs", "network_supply_tariffs"],
    "ტარიფი": ["tariffs", "network_supply_tariffs"],
    "тариф": ["tariffs", "network_supply_tariffs"],
    "regulated": ["tariffs", "general_definitions"],
    "liberalization": ["market_structure", "tariffs"],
    "liberalisation": ["market_structure", "tariffs"],
    "deregulation": ["market_structure", "tariffs"],
    "enguri": ["tariffs", "pso_trading"],
    "vardnili": ["tariffs", "pso_trading"],
    "gardabani": ["tariffs"],
    "gnerc": ["tariffs", "market_structure"],
    "cost-plus": ["tariffs"],
    "capacity fee": ["tariffs", "network_supply_tariffs"],
    "engurhesi": ["tariffs"],

    # Network & end-user supply tariffs (retail side).
    # WARNING: matching below is a bare substring test (`keyword in query_lower`),
    # NOT a word-boundary match. Short keys therefore fire inside unrelated words.
    # This is why the supply company code 'eps' is deliberately absent: it would
    # match "steps", "keeps", "epsilon". Prefer multi-word or distinctive keys.
    "end-user": ["network_supply_tariffs"],
    "end user": ["network_supply_tariffs"],
    "retail tariff": ["network_supply_tariffs"],
    "retail price": ["network_supply_tariffs"],
    "consumer tariff": ["network_supply_tariffs"],
    "consumer price": ["network_supply_tariffs"],
    "household tariff": ["network_supply_tariffs"],
    "household": ["network_supply_tariffs"],
    "per kwh": ["network_supply_tariffs"],
    "gel/kwh": ["network_supply_tariffs"],
    "electricity bill": ["network_supply_tariffs"],
    "distribution tariff": ["network_supply_tariffs"],
    "transmission tariff": ["network_supply_tariffs"],
    "supply tariff": ["network_supply_tariffs"],
    "network tariff": ["network_supply_tariffs"],
    # NOTE: "universal service" is not repeated here -- it already exists in the
    # PSO Trading block above and was extended there. A duplicate literal key
    # would silently shadow the earlier entry (ruff F601).
    "public service provider": ["network_supply_tariffs"],
    "solr": ["network_supply_tariffs"],
    "supplier of last resort": ["network_supply_tariffs"],
    "telasi": ["network_supply_tariffs"],
    "epg": ["network_supply_tariffs"],
    "energo-pro georgia": ["network_supply_tariffs"],
    "voltage": ["network_supply_tariffs"],
    "საბოლოო მომხმარებელი": ["network_supply_tariffs"],
    "конечный потребитель": ["network_supply_tariffs"],

    # PSO Trading
    "pso": ["pso_trading", "tariffs"],
    "public service obligation": ["pso_trading"],
    "telmico": ["pso_trading", "network_supply_tariffs"],
    "ep georgia": ["pso_trading"],
    "procurement": ["pso_trading"],
    "universal service": ["pso_trading", "network_supply_tariffs"],
    "cascade distribution": ["pso_trading"],

    # CfD / PPA
    "cfd": ["cfd_ppa"],
    "contract for difference": ["cfd_ppa"],
    "strike price": ["cfd_ppa"],
    "ppa": ["cfd_ppa"],
    "power purchase agreement": ["cfd_ppa"],
    "support scheme": ["cfd_ppa"],
    "direct contract": ["direct_contracts", "market_structure"],
    "direct contracts": ["direct_contracts", "market_structure"],
    "bilateral contract": ["direct_contracts"],
    "bilateral contracts": ["direct_contracts"],
    "contractual application": ["direct_contracts"],
    "contract registration": ["direct_contracts"],
    "registered contract": ["direct_contracts"],
    "project company": ["direct_contracts", "cfd_ppa"],
    "guaranteed purchase": ["direct_contracts", "cfd_ppa"],
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
    "demand": ["generation_mix"],
    "consumption": ["generation_mix"],
    "გენერაცია": ["generation_mix"],
    "генерация": ["generation_mix"],
    "hydro": ["generation_mix", "balancing_price"],
    "thermal": ["generation_mix", "balancing_price"],
    "wind": ["generation_mix"],
    "solar": ["generation_mix"],
    "generation mix": ["generation_mix"],

    # Trade / import / export
    "import": ["cross_border_trade", "market_structure", "currency_influence"],
    "export": ["cross_border_trade", "market_structure"],
    "trade": ["market_structure"],
    "transit": ["cross_border_trade"],
    "იმპორტი": ["cross_border_trade", "market_structure"],
    "ექსპორტი": ["cross_border_trade", "market_structure"],
    "импорт": ["cross_border_trade", "market_structure"],
    "экспорт": ["cross_border_trade", "market_structure"],
    "interconnection": ["cross_border_trade", "market_structure", "cross_border_capacity"],
    "cross-border": ["cross_border_trade", "market_structure", "cross_border_capacity"],
    "cross border": ["cross_border_trade", "market_structure", "cross_border_capacity"],
    "curtailment": ["cross_border_trade", "cfd_ppa", "balancing_price"],
    "surplus": ["cross_border_trade", "balancing_price"],
    "oversupply": ["balancing_price", "cross_border_trade"],
    "excess supply": ["balancing_price", "cross_border_trade"],
    "atc": ["cross_border_trade"],
    "ntc": ["cross_border_trade"],
    "capacity allocation": ["cross_border_trade"],

    # Interconnection capacity (physical / TTC) — see cross_border_capacity.md.
    # These describe the physical transmission layer (transfer ceilings,
    # operating modes, converter/transformer limits, named interconnections),
    # distinct from the trade RULES in cross_border_trade.md.
    "cross-border capacity": ["cross_border_capacity", "cross_border_trade"],
    "interconnection capacity": ["cross_border_capacity"],
    "transfer capacity": ["cross_border_capacity"],
    "total transfer capacity": ["cross_border_capacity"],
    "ttc": ["cross_border_capacity"],
    "operating mode": ["cross_border_capacity"],
    "synchronous": ["cross_border_capacity"],
    "asynchronous": ["cross_border_capacity"],
    "back-to-back": ["cross_border_capacity"],
    "back to back": ["cross_border_capacity"],
    "hvdc": ["cross_border_capacity"],
    "converter station": ["cross_border_capacity"],
    "synchronous ring": ["cross_border_capacity"],
    "circulating flow": ["cross_border_capacity"],
    "gardabani interconnection": ["cross_border_capacity"],
    "kavkasioni": ["cross_border_capacity"],
    "stepantsminda": ["cross_border_capacity"],
    "mozdok": ["cross_border_capacity"],
    "salkhino": ["cross_border_capacity"],
    "mukhranis": ["cross_border_capacity"],
    "alaverdi": ["cross_border_capacity"],
    "marneuli": ["cross_border_capacity"],
    "ayrum": ["cross_border_capacity"],
    "meskheti": ["cross_border_capacity"],
    "akhaltsikhe": ["cross_border_capacity"],
    "tortum": ["cross_border_capacity"],
    "adjara": ["cross_border_capacity"],
    "black sea interconnection": ["cross_border_capacity"],
    "anaklia": ["cross_border_capacity"],
    "constanta": ["cross_border_capacity"],
    "romania": ["cross_border_capacity"],

    # Market participants
    "esco": ["market_structure"],
    "gse": ["market_structure"],
    "genex": ["exchange_transition", "market_structure"],
    "exchange": ["exchange_transition", "market_structure"],
    "day-ahead": ["exchange_transition", "market_structure"],
    "day ahead": ["exchange_transition", "market_structure"],
    "intraday": ["exchange_transition", "market_structure"],
    "article 17": ["exchange_transition", "market_structure"],
    "17^4": ["exchange_transition"],
    "17⁴": ["exchange_transition"],
    "eligible buyer": ["exchange_transition"],
    "eligible seller": ["exchange_transition"],
    "geostat": ["market_structure"],
    "participant": ["market_structure", "exchange_transition"],

    # Forecasting
    "forecast": ["seasonal_patterns", "balancing_price", "sql_examples"],
    "predict": ["seasonal_patterns", "sql_examples"],
    "projection": ["seasonal_patterns", "sql_examples"],
    "trendline": ["seasonal_patterns", "sql_examples"],
    "პროგნოზი": ["seasonal_patterns", "sql_examples"],
    "прогноз": ["seasonal_patterns", "sql_examples"],

    # Energy security
    "energy security": ["generation_mix", "sql_examples"],
    "import dependency": ["generation_mix", "market_structure"],
    "import dependence": ["generation_mix", "market_structure"],
    "self-sufficiency": ["generation_mix", "sql_examples"],

    # Abkhazeti
    "abkhaz": ["market_structure"],
    "აფხაზეთ": ["market_structure"],

    # Direct customers
    "direct customer": ["market_structure", "exchange_transition", "direct_contracts"],
    "პირდაპირი მომხმარებელი": ["market_structure", "exchange_transition", "direct_contracts"],
    "wholesale market": ["market_structure", "exchange_transition", "direct_contracts"],
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

    relevant = {}
    for stem in sorted(infer_topic_matches(user_query)):
        content = _KNOWLEDGE.get(stem)
        if content:
            relevant[stem] = content
    return json.dumps(relevant, indent=2, ensure_ascii=False)

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


def _direct_topic_matches(user_query: str = "") -> Set[str]:
    """Return only keyword-matched file stems, without a fallback."""
    query_lower = str(user_query or "").lower()
    matched_files: set[str] = set()

    for keyword, file_stems in TOPIC_MAP.items():
        if keyword in query_lower:
            matched_files.update(file_stems)
    return matched_files


def infer_topic_matches(user_query: str = "") -> Set[str]:
    """Infer relevant knowledge file stems directly from query text."""

    matched_files = _direct_topic_matches(user_query)

    if matched_files:
        return matched_files

    query_lower = str(user_query or "").lower()
    definition_patterns = ["what is", "what are", "define", "explain"]
    is_conceptual = any(pattern in query_lower for pattern in definition_patterns)
    return {"general_definitions"} if is_conceptual else {"balancing_price", "sql_examples"}


_MARKDOWN_HEADING_RE = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$")
_SEARCH_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SECTION_SEARCH_STOPWORDS = {
    "about", "after", "also", "and", "are", "before", "but", "can",
    "did", "does", "during", "explain", "for", "from", "have", "how",
    "into", "market", "not", "price", "that", "the", "their", "then",
    "this", "under", "what", "when", "where", "which", "why", "with",
}


def _search_terms(text: str) -> Set[str]:
    """Return small, language-agnostic lexemes for deterministic section ranking."""

    terms: Set[str] = set()
    for raw in _SEARCH_TOKEN_RE.findall(str(text or "").casefold()):
        if raw in _SECTION_SEARCH_STOPWORDS or len(raw) < 2:
            continue
        # A short prefix safely joins common English inflections (drive/drivers,
        # form/formation) while retaining digits and non-Latin search terms.
        terms.add(raw[:4] if len(raw) >= 5 else raw)
    return terms


def _markdown_sections(content: str) -> List[Tuple[int, str, str, str]]:
    """Split Markdown at heading boundaries and retain each section's path."""

    text = str(content or "")
    matches = list(_MARKDOWN_HEADING_RE.finditer(text))
    if not matches:
        return [(0, "", "", text.strip())] if text.strip() else []

    sections: List[Tuple[int, str, str, str]] = []
    stack: List[Tuple[int, str]] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        sections.append((0, "", "", prefix))
    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[match.start():end].strip()
        sections.append((match.start(), " > ".join(item[1] for item in stack), title, section_text))
    return sections


def _normalized_overlap_text(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def _remove_exact_overlap(section: str, exclude_text: str) -> Tuple[str, int]:
    """Remove paragraph-sized verbatim overlap while preserving local context."""

    exclude_normalized = _normalized_overlap_text(exclude_text)
    if not exclude_normalized:
        return section, 0

    kept: List[str] = []
    removed = 0
    for block in re.split(r"\n[ \t]*\n", str(section or "")):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        heading = lines[0] if lines and _MARKDOWN_HEADING_RE.fullmatch(lines[0]) else ""
        body = "\n".join(lines[1:]).strip() if heading else block
        normalized_body = _normalized_overlap_text(body)
        if len(normalized_body) >= 80 and normalized_body in exclude_normalized:
            removed += 1
            if heading:
                kept.append(heading)
            continue
        kept.append(block)
    compacted = "\n\n".join(kept).strip()
    # A heading with no remaining body adds no complementary evidence.
    if compacted and len(compacted.splitlines()) == 1 and _MARKDOWN_HEADING_RE.fullmatch(compacted):
        return "", removed
    return compacted, removed


def _section_score(path: str, title: str, section: str, query_terms: Set[str]) -> int:
    if not query_terms:
        return 1
    heading_terms = _search_terms(f"{path} {title}")
    body_terms = _search_terms(section)
    heading_hits = len(query_terms & heading_terms)
    body_hits = len(query_terms & body_terms)
    return heading_hits * 12 + body_hits * 2


def compact_knowledge_json(
    knowledge_json: str,
    *,
    query: str,
    max_chars: Optional[int] = None,
    exclude_text: str = "",
) -> str:
    """Select relevant Markdown sections and remove exact cross-source overlap.

    The JSON topic keys remain the source-identity contract. Compaction acts only
    when a budget or exclusion corpus is supplied; legacy callers otherwise get
    the byte-identical payload.
    """

    raw = str(knowledge_json or "")
    if max_chars is None and not exclude_text:
        return raw
    if max_chars is not None and len(raw) <= int(max_chars) and not exclude_text:
        return raw
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return raw
    if not isinstance(payload, dict):
        return raw

    budget = max(256, int(max_chars)) if max_chars is not None else max(256, len(raw))
    query_terms = _search_terms(query)
    candidates: List[Tuple[int, int, str, str]] = []
    overlap_blocks_removed = 0
    for stem, content in payload.items():
        if not isinstance(content, str):
            continue
        for position, path, title, section in _markdown_sections(content):
            section, removed = _remove_exact_overlap(section, exclude_text)
            overlap_blocks_removed += removed
            if not section:
                continue
            score = _section_score(path, title, section, query_terms)
            candidates.append((score, -position, str(stem), section))

    # Highest-relevance sections pack first; source keys keep provenance visible.
    candidates.sort(reverse=True)
    selected: Dict[str, List[Tuple[int, str]]] = {}
    for _score, negative_position, stem, section in candidates:
        position = -negative_position
        trial = {
            key: "\n\n".join(text for _position, text in sorted(parts))
            for key, parts in selected.items()
        }
        trial.setdefault(stem, "")
        trial_sections = list(selected.get(stem, [])) + [(position, section)]
        trial[stem] = "\n\n".join(text for _position, text in sorted(trial_sections))
        encoded = json.dumps(trial, indent=2, ensure_ascii=False)
        if len(encoded) <= budget:
            selected.setdefault(stem, []).append((position, section))

    result_payload = {
        stem: "\n\n".join(text for _position, text in sorted(parts))
        for stem, parts in selected.items()
        if parts
    }
    result = json.dumps(result_payload, indent=2, ensure_ascii=False)
    log.info(
        "Knowledge compaction: input_chars=%d output_chars=%d selected_sources=%d "
        "selected_sections=%d overlap_blocks_removed=%d budget=%d",
        len(raw),
        len(result),
        len(result_payload),
        sum(len(parts) for parts in selected.values()),
        overlap_blocks_removed,
        budget,
    )
    return result


def _truncate_brief_source(content: str, budget: int) -> str:
    if len(content) <= budget:
        return content
    marker = "\n...[truncated]"
    if budget <= len(marker):
        return marker[:budget]
    return content[: budget - len(marker)] + marker


def _allocate_brief_source_budgets(
    sources: List[tuple[str, str]],
    total_budget: int,
) -> Dict[str, int]:
    """Split a prompt budget fairly, redistributing unused short-file space."""
    remaining = list(sources)
    remaining_budget = max(0, total_budget)
    allocations: Dict[str, int] = {}
    while remaining:
        share, remainder = divmod(remaining_budget, len(remaining))
        fitting = [
            (stem, content)
            for stem, content in remaining
            if len(content) <= share
        ]
        if not fitting:
            for index, (stem, _content) in enumerate(remaining):
                allocations[stem] = share + (1 if index < remainder else 0)
            break
        fitting_names = {stem for stem, _content in fitting}
        for stem, content in fitting:
            allocations[stem] = len(content)
            remaining_budget -= len(content)
        remaining = [
            (stem, content)
            for stem, content in remaining
            if stem not in fitting_names
        ]
    return allocations


def get_brief_knowledge_for_query(
    user_query: str,
    *,
    max_chars: int = 12000,
) -> str:
    """Return fairly bounded local Markdown context for the cheap Brief path.

    Brief uses direct topic-map matches only, removes SQL examples, and falls
    back to general definitions when no curated topic matches. Standard and
    Report retain their existing broader knowledge-selection behavior.
    """
    matched_files = _direct_topic_matches(user_query)
    matched_files.discard("sql_examples")
    if not matched_files:
        matched_files = {"general_definitions"}

    sources = [
        (stem, _KNOWLEDGE[stem])
        for stem in sorted(matched_files)
        if stem in _KNOWLEDGE
    ]
    if not sources:
        return ""

    char_budget = max(256, int(max_chars))
    separator = "\n\n---\n\n"
    prefixes = {
        stem: f"SOURCE_FILE: {stem}.md\n"
        for stem, _content in sources
    }
    fixed_chars = sum(len(prefixes[stem]) for stem, _content in sources)
    fixed_chars += len(separator) * (len(sources) - 1)
    content_budget = max(0, char_budget - fixed_chars)
    allocations = _allocate_brief_source_budgets(sources, content_budget)
    sections = [
        prefixes[stem] + _truncate_brief_source(content, allocations.get(stem, 0))
        for stem, content in sources
    ]
    result = separator.join(sections)
    log.info(
        "Brief knowledge selected: files=%s context_chars=%d",
        ",".join(stem for stem, _content in sources),
        len(result),
    )
    return result


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
    inferred = infer_topic_matches(fallback_query)
    if preferred == {"general_definitions"} and inferred - {"general_definitions"}:
        preferred = preferred | (inferred - {"general_definitions"})

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
    inferred = infer_topic_matches(fallback_query)
    if preferred == {"general_definitions"} and inferred - {"general_definitions"}:
        preferred = preferred | (inferred - {"general_definitions"})
    if preferred:
        sections = [_KNOWLEDGE[stem] for stem in sorted(preferred) if stem in _KNOWLEDGE]
        return "\n\n---\n\n".join(sections)

    return get_knowledge_for_query(fallback_query)
