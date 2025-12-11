"""
LLM integration and response generation.

Handles:
- Gemini and OpenAI LLM instances (singleton pattern)
- LLM response caching for performance
- Query type classification and focus detection
- SQL generation from natural language
- Answer summarization with domain knowledge
- Domain knowledge filtering and selection
"""
import json
import hashlib
import logging
import time
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    GEMINI_MODEL,
    OPENAI_MODEL,
    MODEL_TYPE
)
from context import DB_SCHEMA_DOC
from domain_knowledge import DOMAIN_KNOWLEDGE
from prompts.few_shot_examples import get_relevant_examples
from utils.metrics import metrics

log = logging.getLogger("Enai")


# -----------------------------
# Cache domain knowledge JSON (once at startup)
# -----------------------------
_DOMAIN_KNOWLEDGE_JSON = json.dumps(DOMAIN_KNOWLEDGE, indent=2)
log.info("✅ Domain knowledge JSON cached at startup")


# -----------------------------
# LLM Response Cache (Phase 1 Optimization)
# -----------------------------

class LLMResponseCache:
    """Simple in-memory cache for LLM responses.

    Phase 1 optimization: Cache identical prompts to avoid repeated LLM calls.
    Future: Migrate to Redis for persistence across restarts.
    """

    def __init__(self, max_size: int = 1000):
        self._cache = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _make_key(self, prompt: str) -> str:
        """Generate cache key from prompt hash."""
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]

    def get(self, prompt: str) -> Optional[str]:
        """Get cached response if exists."""
        key = self._make_key(prompt)
        if key in self._cache:
            self._hits += 1
            log.info(f"✅ LLM cache HIT (hit rate: {self.hit_rate():.1%})")
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, prompt: str, response: str):
        """Cache response for prompt."""
        if len(self._cache) >= self._max_size:
            # Simple LRU: Remove oldest 10% when full
            remove_count = self._max_size // 10
            for _ in range(remove_count):
                self._cache.pop(next(iter(self._cache)))
            log.info(f"🗑️ Cache eviction: removed {remove_count} oldest entries")

        key = self._make_key(prompt)
        self._cache[key] = response

    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate(),
        }


# Global cache instance
llm_cache = LLMResponseCache(max_size=1000)


# -----------------------------
# LLM Instances (Singleton Pattern)
# -----------------------------

_gemini_llm = None
_openai_llm = None


def get_gemini() -> ChatGoogleGenerativeAI:
    """Get cached Gemini LLM instance (singleton pattern).

    Note: convert_system_message_to_human=True is required because Gemini
    doesn't natively support SystemMessages in the LangChain interface.

    Retry configuration: max_retries=2 to prevent quota exhaustion from
    aggressive retry behavior (default is 6 retries with exponential backoff).
    """
    global _gemini_llm
    if _gemini_llm is None:
        _gemini_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            convert_system_message_to_human=True,
            max_retries=2  # Limit retries to prevent quota exhaustion
        )
        log.info("✅ Gemini LLM instance cached (max_retries=2)")
    return _gemini_llm


def get_openai() -> ChatOpenAI:
    """Get cached OpenAI LLM instance (singleton pattern).

    Raises:
        RuntimeError: If OPENAI_API_KEY is not configured
    """
    global _openai_llm
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set (fallback needed)")
    if _openai_llm is None:
        _openai_llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            max_retries=2  # Limit retries to prevent quota exhaustion
        )
        log.info("✅ OpenAI LLM instance cached (max_retries=2)")
    return _openai_llm


# Backward compatibility aliases
make_gemini = get_gemini
make_openai = get_openai


# -----------------------------
# Query Classification Helpers
# -----------------------------

def classify_query_type(user_query: str) -> str:
    """
    Classify query into specific types for better chart/answer decisions.

    Returns:
        - "single_value": One specific value requested
        - "list": Enumeration/listing of items
        - "comparison": Comparing two or more things
        - "trend": Time series analysis
        - "table": Detailed data display
        - "unknown": Cannot determine type
    """
    query_lower = user_query.lower()

    # Single value indicators (highest priority)
    if any(p in query_lower for p in [
        "what is the", "what was the", "how much is", "how much was",
        "რა არის", "რა იყო", "сколько"
    ]) and any(p in query_lower for p in [
        "in june", "in 2024", "for june", "for 2024", "latest", "last month",
        "იუნის", "წელს", "в июне", "в 2024"
    ]):
        return "single_value"

    # List indicators
    if any(p in query_lower for p in [
        "list all", "show all", "enumerate", "which entities",
        "what are the", "name all", "give me all entities",
        "ჩამოთვალე", "ყველა", "перечисли", "какие"
    ]):
        return "list"

    # Comparison indicators
    if any(p in query_lower for p in [
        "compare", " vs ", " vs. ", "versus", "difference between",
        "compared to", "შედარება", "შედარებით", "сравни", "по сравнению"
    ]):
        return "comparison"

    # Trend indicators
    if any(p in query_lower for p in [
        "trend", "over time", "dynamics", "evolution", "change over",
        "from 20", "between 20", "since 20",
        "динамика", "ტენდენცია", "დინამიკა"
    ]):
        return "trend"

    # Table indicators
    if any(p in query_lower for p in [
        "show me all", "give me all", "detailed", "breakdown", "show data",
        "table", "tabular"
    ]):
        return "table"

    return "unknown"


def get_query_focus(user_query: str) -> str:
    """
    Determine the main focus of the query to filter domain knowledge appropriately.

    Returns:
        - "cpi": Consumer Price Index queries
        - "tariff": Tariff-focused queries
        - "generation": Electricity generation queries
        - "balancing": Balancing market/price queries
        - "trade": Import/export/trade queries
        - "general": Cannot determine or multiple focuses
    """
    query_lower = user_query.lower()

    # CPI focus (check first - very specific)
    if any(k in query_lower for k in ["cpi", "inflation", "consumer price index", "ინფლაცია"]):
        return "cpi"

    # Tariff focus (check before balancing - tariff is more specific)
    if any(k in query_lower for k in ["tariff", "ტარიფი", "тариф"]) and \
       not any(k in query_lower for k in ["balancing", "საბალანსო", "баланс"]):
        return "tariff"

    # Generation focus
    if any(k in query_lower for k in ["generation", "generated", "produce", "გენერაცია", "генерация", "производство"]) and \
       not any(k in query_lower for k in ["price", "ფასი", "цена"]):
        return "generation"

    # Trade focus
    if any(k in query_lower for k in ["import", "export", "trade", "იმპორტი", "ექსპორტი", "импорт", "экспорт"]) and \
       not any(k in query_lower for k in ["price", "ფასი", "цена"]):
        return "trade"

    # Balancing focus (check last - most common)
    if any(k in query_lower for k in ["balancing", "p_bal", "საბალანსო", "баланс", "balance market"]):
        return "balancing"

    return "general"


# -----------------------------
# Few-Shot SQL Examples
# -----------------------------

FEW_SHOT_SQL = """
-- Example 1: Monthly average balancing price (USD)
SELECT
  EXTRACT(YEAR FROM date) AS year,
  EXTRACT(MONTH FROM date) AS month,
  AVG(p_bal_usd) AS avg_balancing_usd
FROM price_with_usd
GROUP BY 1,2
ORDER BY 1,2
LIMIT 3750;

-- Example 2: Single-month balancing price (USD)
SELECT p_bal_usd
FROM price_with_usd
WHERE date = '2024-05-01'
LIMIT 3750;

-- Example 3: Generation (thousand MWh) by technology per month
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  type_tech,
  SUM(quantity_tech) AS qty_thousand_mwh
FROM tech_quantity_view
GROUP BY 1,2
ORDER BY 1,2
LIMIT 3750;

-- Example 4: CPI monthly values for electricity fuels
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  cpi
FROM monthly_cpi_mv
WHERE cpi_type = 'electricity_gas_and_other_fuels'
ORDER BY date
LIMIT 3750;

-- Example 5: Balancing price GEL vs shares (no raw quantities)
-- IMPORTANT: Use ILIKE or lowercase comparison for segment to handle different casings
-- Database may contain 'Balancing Electricity', 'balancing', or other variants
-- CRITICAL: Filter entities in denominator to only include relevant balancing entities
WITH shares AS (
  SELECT
    t.date,
    SUM(t.quantity) AS total_qty,
    SUM(CASE WHEN t.entity = 'import' THEN t.quantity ELSE 0 END) AS qty_import,
    SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity ELSE 0 END) AS qty_dereg_hydro,
    SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity ELSE 0 END) AS qty_reg_hpp
  FROM trade_derived_entities t
  WHERE LOWER(REPLACE(t.segment, ' ', '_')) = 'balancing'
    AND t.entity IN ('import', 'deregulated_hydro', 'regulated_hpp',
                     'regulated_new_tpp', 'regulated_old_tpp',
                     'renewable_ppa', 'thermal_ppa')
  GROUP BY t.date
)
SELECT
  TO_CHAR(p.date, 'YYYY-MM') AS month,
  p.p_bal_gel,
  (s.qty_import / NULLIF(s.total_qty,0))      AS share_import,
  (s.qty_dereg_hydro / NULLIF(s.total_qty,0)) AS share_deregulated_hydro,
  (s.qty_reg_hpp / NULLIF(s.total_qty,0))     AS share_regulated_hpp
FROM price_with_usd p
LEFT JOIN shares s ON s.date = p.date
ORDER BY p.date
LIMIT 3750;

-- Example 6: Balancing price (GEL, USD) + tariffs (Enguri, Gardabani, old TPPs) + xrate
WITH tariffs AS (
  SELECT
    d.date,
    (SELECT t1.tariff_gel FROM tariff_with_usd t1 WHERE t1.date = d.date AND t1.entity = 'ltd "engurhesi"1' LIMIT 1) AS enguri_tariff_gel,
    (SELECT t2.tariff_gel FROM tariff_with_usd t2 WHERE t2.date = d.date AND t2.entity = 'ltd "gardabni thermal power plant"' LIMIT 1) AS gardabani_tpp_tariff_gel,
    (SELECT AVG(t3.tariff_gel) FROM tariff_with_usd t3 WHERE t3.date = d.date AND t3.entity IN ('ltd "mtkvari energy"', 'ltd "iec" (tbilresi)', 'ltd "g power" (capital turbines)')) AS grouped_old_tpp_tariff_gel
  FROM price_with_usd d
)
SELECT
  p.date,
  p.p_bal_gel,
  p.p_bal_usd,
  p.xrate,
  tr.enguri_tariff_gel,
  tr.gardabani_tpp_tariff_gel,
  tr.grouped_old_tpp_tariff_gel
FROM price_with_usd p
LEFT JOIN tariffs tr ON tr.date = p.date
ORDER BY p.date
LIMIT 3750;

-- Example 7: Summer vs Winter averages
WITH seasons AS (
  SELECT date,
         CASE WHEN EXTRACT(MONTH FROM date) IN (4,5,6,7) THEN 'summer' ELSE 'winter' END AS season
  FROM price_with_usd
)
SELECT
  s.season,
  AVG(p.p_bal_gel) AS avg_bal_price_gel,
  AVG(tr.enguri_tariff_gel) AS avg_enguri_tariff_gel
FROM seasons s
JOIN price_with_usd p ON p.date = s.date
JOIN (
  SELECT date, tariff_gel AS enguri_tariff_gel
  FROM tariff_with_usd WHERE entity = 'ltd "engurhesi"1'
) tr ON tr.date = s.date
GROUP BY s.season
ORDER BY s.season
LIMIT 3750;

-- Example 8: Renewable PPA share in balancing electricity for specific month
-- CRITICAL: Always use LOWER(REPLACE(segment, ' ', '_')) for segment filtering
-- This example shows how to calculate share of a specific entity
WITH shares AS (
  SELECT
    date,
    SUM(quantity) AS total_qty,
    SUM(CASE WHEN entity = 'renewable_ppa' THEN quantity ELSE 0 END) AS qty_renewable_ppa,
    SUM(CASE WHEN entity = 'thermal_ppa' THEN quantity ELSE 0 END) AS qty_thermal_ppa,
    SUM(CASE WHEN entity = 'import' THEN quantity ELSE 0 END) AS qty_import
  FROM trade_derived_entities
  WHERE LOWER(REPLACE(segment, ' ', '_')) = 'balancing'
  GROUP BY date
)
SELECT
  date,
  (qty_renewable_ppa / NULLIF(total_qty, 0)) AS share_renewable_ppa,
  (qty_thermal_ppa / NULLIF(total_qty, 0)) AS share_thermal_ppa,
  (qty_import / NULLIF(total_qty, 0)) AS share_import
FROM shares
WHERE date = '2024-06-01'
ORDER BY date
LIMIT 3750;

-- Example 9: Simple entity list (NO price context needed)
SELECT DISTINCT entity
FROM trade_derived_entities
ORDER BY entity
LIMIT 3750;

-- Example 10: Single tariff value query (NO balancing context needed)
SELECT tariff_gel
FROM tariff_with_usd
WHERE entity = 'ltd "engurhesi"1'
  AND date = '2024-06-01'
LIMIT 1;

-- Example 11: Generation by technology (NO price context needed)
SELECT
  type_tech,
  SUM(quantity_tech) AS total_generation_thousand_mwh
FROM tech_quantity_view
WHERE EXTRACT(YEAR FROM date) = 2023
GROUP BY type_tech
ORDER BY total_generation_thousand_mwh DESC
LIMIT 3750;

-- Example 12: CPI trend (NO electricity price context needed)
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  cpi AS electricity_fuels_cpi
FROM monthly_cpi_mv
WHERE cpi_type = 'electricity_gas_and_other_fuels'
  AND date >= '2023-01-01'
ORDER BY date
LIMIT 3750;

-- Example 13: Tariff comparison (NO balancing price context needed)
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  entity,
  tariff_gel
FROM tariff_with_usd
WHERE entity IN ('ltd "engurhesi"1', 'ltd "gardabni thermal power plant"')
  AND date >= '2024-01-01'
ORDER BY date, entity
LIMIT 3750;

-- ============================================================================
-- AGGREGATION EXAMPLES (CRITICAL for Total vs Breakdown disambiguation)
-- ============================================================================

-- Example A1: TOTAL generation (single number, all technologies)
-- User: "What was total generation in 2023?"
-- Intent: Single total across ALL technologies
SELECT
  SUM(quantity_tech) * 1000 AS total_generation_mwh
FROM tech_quantity_view
WHERE EXTRACT(YEAR FROM date) = 2023
  AND type_tech IN ('hydro', 'thermal', 'wind', 'solar')
LIMIT 3750;
-- IMPORTANT: NO GROUP BY - returns single row

-- Example A2: TOTAL generation BY TECHNOLOGY (breakdown)
-- User: "What was total generation by technology in 2023?"
-- Intent: Total for EACH technology
SELECT
  type_tech,
  SUM(quantity_tech) * 1000 AS total_generation_mwh
FROM tech_quantity_view
WHERE EXTRACT(YEAR FROM date) = 2023
  AND type_tech IN ('hydro', 'thermal', 'wind', 'solar')
GROUP BY type_tech
ORDER BY total_generation_mwh DESC
LIMIT 3750;
-- IMPORTANT: Has GROUP BY - returns multiple rows (one per technology)

-- Example A3: AVERAGE balancing price (single number)
-- User: "What was average balancing price in 2023?"
-- Intent: Single average across entire year
SELECT
  AVG(p_bal_gel) AS average_balancing_price_gel
FROM price_with_usd
WHERE EXTRACT(YEAR FROM date) = 2023
LIMIT 3750;
-- IMPORTANT: NO GROUP BY - returns single average

-- Example A4: SHARE calculation (percentage breakdown)
-- User: "What is share of each technology in total generation for 2023?"
-- Intent: Percentage contribution of each technology
WITH totals AS (
  SELECT
    type_tech,
    SUM(quantity_tech) AS tech_total
  FROM tech_quantity_view
  WHERE EXTRACT(YEAR FROM date) = 2023
    AND type_tech IN ('hydro', 'thermal', 'wind', 'solar')
  GROUP BY type_tech
),
grand_total AS (
  SELECT SUM(tech_total) AS overall_total FROM totals
)
SELECT
  t.type_tech,
  t.tech_total * 1000 AS generation_mwh,
  gt.overall_total * 1000 AS total_generation_mwh,
  ROUND((t.tech_total / gt.overall_total) * 100, 2) AS share_percent
FROM totals t, grand_total gt
ORDER BY share_percent DESC
LIMIT 3750;
-- IMPORTANT: Uses CTE to calculate shares properly
"""


# -----------------------------
# Domain Knowledge Selection
# -----------------------------

def get_relevant_domain_knowledge(user_query: str, use_cache: bool = True) -> str:
    """Return domain knowledge JSON, filtered by query focus to reduce token usage.

    Args:
        user_query: The user's query text
        use_cache: If True, use full cached JSON. If False, select relevant sections only.

    Returns:
        JSON string of domain knowledge (full or filtered)

    This function can reduce token usage by 50-70% when use_cache=False by including
    only sections relevant to the query focus area.
    """
    if use_cache:
        return _DOMAIN_KNOWLEDGE_JSON

    query_lower = user_query.lower()
    relevant = {}

    # ------------------------------------------------------------------
    # Keyword triggers – each key must exist in DOMAIN_KNOWLEDGE
    # ------------------------------------------------------------------
    triggers = {
        "BalancingPriceDrivers": [
            "balancing", "price", "p_bal", "cost", "driver", "why", "increase", "decrease",
            "xrate", "share", "composition", "hydro", "thermal", "import", "ppa"
        ],
        "BalancingPriceFormation": [
            "weighted", "average", "weighting", "entity", "segment", "balancing_electricity"
        ],
        "BalancingMarketStructure": [
            "balancing market", "imbalance", "settlement", "esco", "brp", "monthly"
        ],
        "BalancingPriceDecomposition": [
            "decomposition", "contribution", "share_change", "entity contribution"
        ],
        "BalancingMarketLogic": [
            "deviation", "forecast", "actual", "residual", "mix"
        ],
        "TariffStructure": [
            "tariff", "regulated", "enguri", "vardnili", "gardabani", "tpp", "cost-plus",
            "gnerc", "capacity fee"
        ],
        "TariffDependencies": [
            "enguri", "gardabani", "old tpp", "mtkvar", "g-power"
        ],
        "tariff_entities": [
            "engurhesi", "energo-pro", "dzevruli", "gumati", "shaori", "rioni",
            "lajanuri", "zhinvali", "khrami", "mtkvari energy", "tbilisi tpp"
        ],
        "price_with_usd": [
            "p_bal_gel", "p_bal_usd", "p_dereg_gel", "p_dereg_usd", "usd"
        ],
        "CurrencyInfluence": [
            "gel", "usd", "exchange rate", "depreciation", "xrate"
        ],
        "SeasonalityPatterns": [
            "summer", "winter", "april", "july", "august", "march", "season"
        ],
        "SeasonalTrends": [
            "seasonal", "cagr", "trend", "hydro dominant", "thermal dominant"
        ],
        "CfD_Contracts": [
            "cfd", "contract for difference", "strike price", "renewable ppa", "central dispatch"
        ],
        "RenewableIntegration": [
            "renewable", "ppa", "solar", "wind", "integration"
        ],
        "ImportDependence": [
            "import", "export", "dependence", "turkey", "azeri"
        ],
        "TransmissionNetworkDevelopment": [
            "transmission", "tyndp", "gse", "west-east", "congestion", "substation"
        ],
        "GenerationAdequacyAndForecast": [
            "adequacy", "forecast", "capacity", "plexos", "2034"
        ],
        "MarketParticipantsAndDataSources": [
            "gnerc", "esco", "gse", "genex", "geostat", "participant"
        ],
        "DataEvidenceIntegration": [
            "evidence", "view", "materialized", "chart", "cpi"
        ],
        "TableSelectionGuidance": [
            "table", "view", "tech_quantity", "trade_derived", "which table", "what table"
        ],
        "EnergySecurityAnalysis": [
            "energy security", "import dependence", "self-sufficiency", "უსაფრთხოება",
            "დამოკიდებულება", "თვითკმარობა", "local generation", "domestic generation",
            "independence", "vulnerability", "მოწყვლადობა"
        ],
        "PriceComparisonRules": [
            "price trend", "price comparison", "price increase", "price decrease",
            "compare price", "ფასის ტენდენცია", "ფასის ზრდა", "how much price"
        ]
    }

    for section, keywords in triggers.items():
        if any(k in query_lower for k in keywords):
            if section in DOMAIN_KNOWLEDGE:
                relevant[section] = DOMAIN_KNOWLEDGE[section]

    # ------------------------------------------------------------------
    # FALLBACK: always include BalancingPriceDrivers (core for price questions)
    # ------------------------------------------------------------------
    if not relevant:
        relevant = {
            "note": "No specific domain knowledge matched the query.",
            "BalancingPriceDrivers": DOMAIN_KNOWLEDGE.get("BalancingPriceDrivers", {})
        }

    return json.dumps(relevant, indent=2)


# -----------------------------
# SQL Generation
# -----------------------------

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
def llm_generate_plan_and_sql(
    user_query: str,
    analysis_mode: str,
    lang_instruction: str = "Respond in English.",
    domain_reasoning: str = ""  # Deprecated - kept for backward compatibility
) -> str:
    """
    Generate analytical plan and SQL query from natural language.

    Uses LLM to:
    1. Analyze user intent
    2. Generate structured plan (JSON)
    3. Generate PostgreSQL query

    Args:
        user_query: Natural language query
        analysis_mode: "analyst" for trend analysis, "general" for basic queries
        lang_instruction: Language instruction for LLM
        domain_reasoning: Deprecated parameter (kept for compatibility)

    Returns:
        Combined output: "{plan_json}---SQL---{sql_query}"

    Raises:
        Exception: If both Gemini and OpenAI fail
    """
    # Phase 1C Optimization: Merged domain reasoning into this call
    # Check cache first (cache key no longer includes domain_reasoning since it's internal now)
    cache_input = f"sql_generation_v2|{user_query}|{analysis_mode}|{lang_instruction}"
    cached_response = llm_cache.get(cache_input)
    if cached_response:
        log.info("📝 Plan/SQL: (cached)")
        return cached_response

    # Phase 1C: Include domain reasoning as internal step
    system = (
        "You are an analytical PostgreSQL generator for Georgian energy market data. "
        "Your task is to perform FOUR steps internally, then output plan + SQL: "
        "\n"
        "**STEP 1 (Internal - Analyze Intent):** "
        "Think like an energy market analyst. What is the user really asking? "
        "What domain concepts are involved (price drivers, composition, exchange rates, seasonal patterns)? "
        "What metrics and time periods are needed? "
        "\n"
        "**STEP 2 (Internal - Chart Strategy):** "
        "Analyze data dimensions and decide chart organization. "
        "NEVER mix different dimensions on the same chart: "
        "- Don't mix % (shares) with prices (GEL/USD) "
        "- Don't mix prices with quantities (MWh) "
        "- Don't mix exchange rate (xrate) with prices or shares "
        "- Don't mix different units (GEL vs USD vs % vs MWh) "
        "If query involves multiple dimensions → create separate chart groups. "
        "Chart types: 'line' for trends, 'bar' for comparisons, 'stacked_bar' or 'stacked_area' for composition/shares. "
        "\n"
        "**STEP 3 (Output - Plan):** "
        "Extract the analysis intent, target variables, period, AND chart strategy as JSON. "
        "\n"
        "**STEP 4 (Output - SQL):** "
        "Write a single, correct PostgreSQL SELECT query to fulfill the plan. "
        "\n"
        "Rules: no INSERT/UPDATE/DELETE; no DDL; NO comments; NO markdown fences. "
        "Use only documented tables and columns. Prefer monthly aggregation. "
        "If USD prices are requested, prefer price_with_usd / tariff_with_usd views. "
        "CRITICAL: Always use ENGLISH column aliases in SQL output (e.g., AS month, AS balancing_price_gel), "
        "never use Georgian/Russian names in column aliases, even if the user query is in Georgian/Russian. "
        f"{lang_instruction}"
    )

    # Use selective domain knowledge to reduce tokens (30-40% savings)
    domain_json = get_relevant_domain_knowledge(user_query, use_cache=False)

    plan_format = {
        "intent": "trend_analysis" if analysis_mode == "analyst" else "general",
        "target": "<metric name>",
        "period": "YYYY-YYYY or YYYY-MM to YYYY-MM",
        "chart_strategy": "single or multiple",
        "chart_groups": [
            {
                "type": "line or bar or stacked_bar or stacked_area",
                "metrics": ["column_name1", "column_name2"],
                "title": "Chart title",
                "y_axis_label": "Unit (e.g., GEL/MWh, %, thousand MWh)"
            }
        ]
    }

    # Build guidance dynamically based on query focus
    query_focus = get_query_focus(user_query)
    query_lower = user_query.lower()

    guidance_sections = []

    # Always include basic rules
    guidance_sections.append("- Use ONLY documented materialized views.")
    guidance_sections.append("- Aggregation default = monthly. For energy_balance_long_mv, use yearly.")
    guidance_sections.append("- When USD values appear, *_usd = *_gel / xrate.")

    # Always include chart strategy rules
    guidance_sections.append("""
CHART STRATEGY RULES (CRITICAL):
- NEVER mix dimensions on same chart: % vs GEL vs MWh vs xrate must be separate
- Example 1: If query asks for "price and shares" → create 2 chart groups:
  * Group 1: price (GEL/MWh) - line chart
  * Group 2: shares (%) - stacked_area or stacked_bar
- Example 2: If query asks for "price and exchange rate" → create 2 chart groups:
  * Group 1: balancing_price_gel (GEL/MWh) - line chart
  * Group 2: xrate (GEL/USD) - line chart
- Example 3: If query asks for "generation composition" → single chart:
  * Group 1: share_hydro, share_thermal, share_wind (%) - stacked_area
- Chart types:
  * 'line' for price trends, exchange rate trends
  * 'bar' for entity comparisons, monthly comparisons
  * 'stacked_bar' or 'stacked_area' for composition (shares, generation mix)
- Max 5 metrics per chart group to avoid clutter
""")

    # Conditionally include balancing-specific guidance
    if query_focus == "balancing" or any(k in query_lower for k in ["balancing", "p_bal", "საბალანსო"]):
        guidance_sections.append("""
BALANCING PRICE ANALYSIS:
- Weighted-average balancing price = weighted by total balancing-market quantities
- Entities: deregulated_hydro, import, regulated_hpp, regulated_new_tpp, regulated_old_tpp, renewable_ppa, thermal_ppa
- PRIMARY DRIVER #1: xrate (exchange rate) - MOST IMPORTANT for GEL/MWh price
  * Use xrate from price_with_usd view
  * Critical because gas and imports are USD-priced
- PRIMARY DRIVER #2: Composition (shares) - CRITICAL for both GEL and USD prices
  * Calculate shares from trade_derived_entities
  * IMPORTANT: Use LOWER(REPLACE(segment, ' ', '_')) = 'balancing' for segment filter
  * Use share CTE pattern, no raw quantities
  * Higher cheap source shares (regulated HPP, deregulated hydro) → lower prices
  * Higher expensive source shares (import, thermal PPA) → higher prices
- For seasonal analysis: Summer (Apr–Jul) has lower prices due to hydro generation
""")

    # Conditionally include seasonal guidance
    if any(k in query_lower for k in ["season", "summer", "winter", "сезон", "ზაფხულ", "ზამთარ"]):
        guidance_sections.append("- Season is a derived dimension: use CASE WHEN EXTRACT(MONTH FROM date) IN (4,5,6,7) THEN 'summer' ELSE 'winter' END AS season")

    # Conditionally include tariff guidance
    if query_focus == "tariff" or any(k in query_lower for k in ["tariff", "ტარიფი", "тариф"]):
        guidance_sections.append("""
TARIFF ANALYSIS:
- Key entities: Enguri ('ltd "engurhesi"1'), Gardabani TPP ('ltd "gardabni thermal power plant"')
- Thermal tariffs depend on gas price (USD) → correlated with xrate
- Use tariff_with_usd view for tariff queries
""")

    # Conditionally include CPI guidance
    if query_focus == "cpi" or any(k in query_lower for k in ["cpi", "inflation", "ინფლაცია"]):
        guidance_sections.append("- CPI data: use monthly_cpi_mv, filter by cpi_type = 'electricity_gas_and_other_fuels'")

    guidance = "\n".join(guidance_sections)

    # Phase 1C Fix: Use selective example loading to reduce token usage
    # Load only 2 relevant example categories (~800-1,500 tokens instead of ~5,800)
    # This keeps domain knowledge prominent and restores detailed answer quality
    relevant_examples = get_relevant_examples(user_query, max_categories=2)

    # Phase 1C: Prompt structure updated - domain reasoning is now internal
    prompt = f"""
User question:
{user_query}

Domain knowledge (for Step 1 internal reasoning):
{domain_json}

Schema:
{DB_SCHEMA_DOC}

Guidance:
{guidance}

Examples (Few-Shot Learning - Study these patterns):
{relevant_examples}

Additional SQL Syntax Examples:
{FEW_SHOT_SQL}

Output Format:
Return a single string containing two parts, separated by '---SQL---'. The first part is a JSON object (the plan), and the second part is the raw SELECT statement.

Example Output:
{json.dumps(plan_format)}
---SQL---
SELECT ...
"""
    llm_start = time.time()
    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        combined_output = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        metrics.log_llm_call(time.time() - llm_start)
    except Exception as e:
        log.warning(f"Combined generation failed: {e}")
        # Fallback to OpenAI if Gemini fails
        try:
            llm = make_openai()
            combined_output = llm.invoke([("system", system), ("user", prompt)]).content.strip()
            metrics.log_llm_call(time.time() - llm_start)
        except Exception as e_f:
             log.warning(f"Combined generation failed with fallback: {e_f}")
             metrics.log_error()
             raise e_f # Re-raise final exception

    # Phase 1B Optimization: Cache the response
    llm_cache.set(cache_input, combined_output)

    return combined_output


# -----------------------------
# Answer Summarization
# -----------------------------

def llm_summarize(user_query: str, data_preview: str, stats_hint: str, lang_instruction: str = "Respond in English.") -> str:
    """
    Generate analytical summary from data and statistics.

    Uses LLM to create concise, domain-aware answers based on query results.

    Args:
        user_query: Original user query
        data_preview: Preview of query results
        stats_hint: Statistical summary of results
        lang_instruction: Language instruction for response

    Returns:
        Natural language summary

    Raises:
        Exception: If both Gemini and OpenAI fail
    """
    # Phase 1 Optimization: Check cache first
    # Create cache key from all inputs
    cache_input = f"{user_query}|{data_preview}|{stats_hint}|{lang_instruction}"
    cached_response = llm_cache.get(cache_input)
    if cached_response:
        return cached_response

    system = (
        "Provide a DETAILED analytical answer based on the data preview and statistics. "
        "Use domain knowledge to explain causality and mechanisms. "
        "Do NOT introduce yourself or include greetings - answer the question directly.\n\n"

        "OUTPUT FORMAT BY QUERY TYPE:\n\n"

        "FOR PRICE DRIVER / CORRELATION QUERIES:\n"
        "**[Topic]: ანალიტიკური შეჯამება** (Bold header)\n\n"
        "[Opening paragraph with key finding]\n\n"
        "1. **[First Factor]:** (Bold, numbered)\n"
        "   - [Detailed explanation with ACTUAL DATA VALUES from data preview]\n"
        "   - [Cite correlation if available in stats_hint: e.g., 'კორელაცია -0.66']\n"
        "   - [Explain mechanism/causality using domain knowledge]\n\n"
        "2. **[Second Factor]:** (Bold, numbered)\n"
        "   - [Detailed explanation with ACTUAL DATA VALUES from data preview]\n"
        "   - [Cite correlation if available in stats_hint: e.g., 'კორელაცია 0.61']\n"
        "   - [Explain mechanism/causality using domain knowledge]\n\n"

        "FOR SIMPLE QUERIES (single value, list):\n"
        "- Direct answer (1-2 sentences with numbers and units)\n"
        "- Brief context if relevant\n\n"

        "MANDATORY REQUIREMENTS:\n"
        "- If stats_hint contains correlation coefficients → YOU MUST cite them explicitly\n"
        "- If data preview shows share_* columns → cite ACTUAL VALUES (e.g., '22% to 35%'), not generic statements\n"
        "- For price analysis: Start with composition (share changes) using SPECIFIC NUMBERS from data\n"
        "- Use bold headers (**text**) and numbered points (1., 2.) for structured analysis\n"
        "- NO hedging language when you have data (no 'probably', 'სავარაუდოდ', 'შესაძლოა')\n\n"

        "FORMATTING RULES:\n"
        "- Numbers: Use thousand separators (1,234 not 1234)\n"
        "- Percentages: One decimal place (15.3% not 15.27% or 15%)\n"
        "- Units: ALWAYS include (thousand MWh, GEL/MWh, %, GEL/USD)\n"
        "- Prices: ALWAYS separate summer (April-July) and winter (Aug-Mar)\n"
        "- Never use raw column names (use 'balancing price in GEL' not 'p_bal_gel')\n\n"

        "EXAMPLE EXCELLENT OUTPUT (price driver query in Georgian):\n"
        "**საბალანსო ელექტროენერგიის ფასზე მოქმედი ფაქტორები: ანალიტიკური შეჯამება**\n\n"
        "საბალანსო ელექტროენერგიის ფასს ძირითადად ორი მთავარი ფაქტორი განსაზღვრავს: გენერაციის სტრუქტურა და ლარის გაცვლითი კურსი.\n\n"
        "1. **გენერაციის სტრუქტურა:** ფასი პირდაპირ არის დამოკიდებული იმაზე, თუ რომელი წყაროებიდან "
        "(ჰესი, თესი, იმპორტი) მიეწოდება ენერგია ბაზარს. როდესაც მიწოდებაში მაღალია იაფი რესურსის, "
        "მაგალითად, რეგულირებული ჰესების წილი, საბალანსო ფასი მცირდება. სტატისტიკურად, რეგულირებული "
        "ჰესების წილს ფასთან ძლიერი უარყოფითი კორელაცია აქვს (-0.66). როდესაც იზრდება ძვირადღირებული "
        "წყაროების, როგორიცაა იმპორტი და თბოსადგურები, წილი, ფასი იმატებს.\n\n"
        "2. **გაცვლითი კურსი (GEL/USD):** ეს ფაქტორი კრიტიკულად მნიშვნელოვანია ლარში დენომინირებული "
        "ფასისთვის. ვინაიდან თბოსადგურების საწვავი (ბუნებრივი აირი) და იმპორტირებული ელექტროენერგია "
        "დოლარში იძენება, ლარის გაუფასურება (კურსის ზრდა) პირდაპირ აისახება საბალანსო ენერგიის ფასის "
        "ზრდაზე. კორელაციის ანალიზი აჩვენებს ძლიერ დადებით კავშირს (0.61) გაცვლით კურსსა და საბალანსო "
        "ფასს შორის.\n\n"

        f"{lang_instruction}"
    )

    # Phase 1 Optimization: Determine query complexity for conditional guidance
    query_type = classify_query_type(user_query)
    query_focus = get_query_focus(user_query)
    query_lower = user_query.lower()

    # Simple queries don't need extensive domain knowledge or guidance
    needs_full_guidance = query_type not in ["single_value", "list"]

    # Use selective domain knowledge to reduce tokens (30-40% savings)
    # Phase 1 Optimization: Skip domain knowledge for simple queries
    if needs_full_guidance:
        domain_json = get_relevant_domain_knowledge(user_query, use_cache=False)
        log.info(f"📚 Using full domain knowledge for {query_type} query")
    else:
        domain_json = "{}"  # Minimal for simple queries
        log.info(f"📚 Skipping domain knowledge for {query_type} query (optimization)")

    # Build guidance dynamically based on query focus
    guidance_sections = []

    # Always include focus rules at the top
    guidance_sections.append("""
IMPORTANT RULES - STAY FOCUSED:
1. Answer ONLY what the user asked - don't discuss unrelated topics
2. If query is about CPI/inflation → discuss CPI only (not electricity prices unless comparing affordability)
3. If query is about tariffs → discuss tariffs only (not balancing prices)
4. If query is about generation/quantities → discuss generation only (not prices)
5. If query is about entities/list → provide the list only (no price analysis)
6. Only discuss balancing price if explicitly asked or if query contains balancing price keywords
7. Keep answers concise (1-3 sentences) unless detailed analysis requested

CRITICAL: NEVER use raw database column names in your answer
❌ WRONG: "share_hydro increased", "p_bal_gel rose", "tariff_gel changed"
✅ CORRECT: "hydro generation share increased", "balancing price in GEL rose", "tariff in GEL changed"
Always use descriptive, natural language terms regardless of response language.
""")

    # Add seasonal statistics guidance if stats_hint contains seasonal analysis
    if "SEASONAL-ADJUSTED TREND ANALYSIS" in stats_hint:
        guidance_sections.append("""
CRITICAL: SEASONAL-ADJUSTED TREND ANALYSIS RULES
The stats_hint contains seasonal-adjusted statistics (year-over-year growth, CAGR, etc.).
These are ALREADY ADJUSTED for seasonality - use them directly!

MANDATORY RULES:
1. Use the "Overall growth" percentage from stats_hint for multi-year trends
   - DO NOT compare first month to last month directly
   - DO NOT say "doubled" or "tripled" based on raw monthly data
   - USE the calculated CAGR (average annual growth rate)

2. Pay attention to incomplete year warnings
   - If stats say "Last year has only X months" → mention this caveat
   - DO NOT treat incomplete years as full years in trend analysis

3. For trend queries:
   - Cite the year range: "From [first_year] to [last_year]"
   - Cite the overall growth: "increased by [overall_growth_pct]%"
   - Cite the CAGR: "average annual growth of [cagr]%"
   - Mention seasonal pattern if relevant: "peak in [peak_month], low in [low_month]"

4. Distinguish between:
   - Long-term trend (use CAGR from stats)
   - Seasonal variation (use peak_month/low_month from stats)
   - Recent momentum (use recent_12m_growth if available)

EXAMPLES:
✅ CORRECT: "From 2015 to 2023, demand increased by 25.5% (overall growth), with an average annual growth rate of 3.2% (CAGR). Demand shows strong seasonality, peaking in January (winter) and reaching lows in July (summer)."

❌ WRONG: "Demand almost doubled from 171k MWh to 313k MWh" (this compares January to August - pure seasonality!)

If seasonal stats are present, they are the AUTHORITATIVE source for trends. Trust them over raw data patterns.
""")

    # Phase 1 Optimization: Only include heavy guidance for complex queries
    # Simple queries (single_value, list) skip balancing/tariff/CPI guidance
    # Conditionally include balancing-specific guidance
    if needs_full_guidance and (query_focus == "balancing" or any(k in query_lower for k in ["balancing", "p_bal", "балансовая", "ბალანსის"])):
        guidance_sections.append("""
CRITICAL ANALYSIS GUIDELINES for balancing electricity price:

⚠️ MANDATORY RULES - NO EXCEPTIONS:

1. **CITE ACTUAL NUMBERS FROM DATA PREVIEW** - This is the most important rule:

   STEP-BY-STEP PROCESS:
   a) Look at data preview - find the rows for the periods being compared
   b) Extract EXACT percentage values for share_* columns
   c) Format as: "წილი გაიზარდა/შემცირდა X%-დან Y%-მდე"

   EXAMPLES:
   - ✅ CORRECT: "რეგულირებული ჰესების წილი გაიზარდა 22.3%-დან 35.7%-მდე"
   - ✅ CORRECT: "იმპორტის წილი შემცირდა 18.5%-დან 8.2%-მდე"
   - ❌ WRONG: "ჰიდროგენერაციის წილი გაიზარდა" (no specific numbers!)
   - ❌ WRONG: "რეგულირებული ჰესების მაღალი წილი" (which period? what value?)

   Then explain price impact:
   - ✅ "რადგან რეგულირებული ჰესები იაფია, ფასი შემცირდა"
   - ✅ "რადგან იმპორტი ძვირია, ფასი გაიზარდა"

2. **FOR MONTH-TO-MONTH COMPARISONS**:
   - Find April row and May row in data preview
   - Compare each share_* value between the two months
   - Cite at least 2-3 main changes with exact numbers
   - Focus on largest changes that explain price movement

3. **FOR LONG-TERM TRENDS** (multi-year or annual):
   - MANDATORY: Separate summer (April-July) vs winter (Aug-March) analysis
   - Calculate average shares for summer months
   - Calculate average shares for winter months
   - Explain composition differences:
     * Summer: Higher hydro share (cheap) → lower prices
     * Winter: Higher thermal/import share (expensive) → higher prices
   - Cite specific percentage changes for each season

4. **USE CORRELATION DATA**: If stats_hint contains correlation coefficients, YOU MUST cite them
   - Example: "კორელაცია -0.66 რეგულირებულ ჰესებსა და ფასს შორის"
   - Example: "კორელაცია 0.61 გაცვლით კურსსა და ფასს შორის"
   - NEVER say "probably" when you have correlation proving causality

5. **NO HEDGING LANGUAGE** when you have data:
   - ❌ FORBIDDEN: "სავარაუდოდ" (probably), "შესაძლოა" (possibly), "ალბათ" (perhaps)
   - ✅ REQUIRED: "იმის გამო, რომ" (because), "რაც გამოწვეულია" (which is caused by)

6. **STRUCTURED ANALYSIS FORMAT**:

   **[Question topic]: ანალიტიკური შეჯამება**

   [Opening: state overall price change with numbers]

   1. **გენერაციის სტრუქტურა (Composition):**
      - [List 2-3 main share changes with EXACT numbers from data]
      - [Explain: cheap sources (regulated HPP, deregulated hydro) vs expensive (import, thermal PPA, renewable PPA)]
      - [Cite correlation if available]
      - [For long-term: compare summer vs winter composition]

   2. **გაცვლითი კურსი (Exchange Rate):**
      - [Cite actual xrate change from data: from X to Y GEL/USD]
      - [Explain: gas and imports priced in USD]
      - [Cite correlation if available]

PRICE LEVEL GUIDANCE (use when explaining why sources are cheap/expensive):
- Cheap sources: Regulated HPP (~30-40 GEL/MWh), Deregulated hydro (~40-50 GEL/MWh)
- Expensive sources: Import (market-based), Thermal PPA (market-based), Renewable PPA (market-based)
- Note: DO NOT disclose specific PPA/import price estimates - just say "market-based" or "expensive"

PRIMARY DRIVERS (in order of importance):
1. Composition (shares of entities) - MUST cite actual numbers from data
2. Exchange Rate (xrate) - MUST cite actual change from data
3. Seasonal patterns - MUST separate summer/winter for long-term trends

CONFIDENTIALITY RULES:
- DO disclose: regulated tariffs (~30-40 GEL/MWh), deregulated hydro prices (~40-50 GEL/MWh), correlations
- DO NOT disclose: specific PPA price estimates, specific import price estimates
- When discussing expensive sources: say "market-based" without numbers
""")

    # Conditionally include tariff-specific guidance (only for complex queries)
    if needs_full_guidance and (query_focus == "tariff" or any(k in query_lower for k in ["tariff", "тариф", "ტარიფ"])):
        guidance_sections.append("""
TARIFF ANALYSIS GUIDELINES:
- Tariffs follow GNERC-approved cost-plus methodology.
- Thermal tariffs include a Guaranteed Capacity Fee (fixed) plus a variable per-MWh cost based on gas price and efficiency.
- Gas is priced in USD, so thermal tariffs correlate with the GEL/USD exchange rate (xrate).
- Do not apply seasonal logic to tariff analyses.
- Focus on annual or multi-year trends explained by regulatory cost-plus principles: fixed guaranteed-capacity fee, variable gas-linked component, and exchange-rate sensitivity.
""")

    # Conditionally include CPI-specific guidance (only for complex queries)
    if needs_full_guidance and query_focus == "cpi":
        guidance_sections.append("""
CPI ANALYSIS GUIDELINES:
- Focus on CPI category 'electricity_gas_and_other_fuels' trends.
- When comparing to electricity prices (tariff_gel or p_bal_gel), frame as affordability comparison.
- Describe CPI trend direction, magnitude, and time periods clearly.
- Only discuss electricity prices if user asks for affordability comparison.
""")

    # Conditionally include generation-specific guidance (only for complex queries)
    if needs_full_guidance and query_focus == "generation":
        guidance_sections.append("""
GENERATION ANALYSIS GUIDELINES:
- Focus on quantities (thousand_mwh) by technology type or entity.
- Describe generation trends, shares, and seasonal patterns.
- Summer vs Winter comparison relevant for hydro vs thermal generation.
- Only discuss prices if user explicitly asks about price-generation relationships.

CRITICAL: ENERGY SECURITY AND IMPORT DEPENDENCE:
- Thermal generation uses imported natural gas and is NOT fully domestic/independent
- True local/independent generation: Hydro, Wind, Solar (no fuel imports)
- Import-dependent generation: Thermal (imported gas) + Direct electricity import
- When discussing energy security: "Winter import dependence includes both direct electricity imports AND gas imports for thermal generation"
- ❌ NEVER say "thermal reduces import dependence" or "thermal is local generation"
- ✅ ALWAYS clarify "thermal relies on imported gas" when discussing energy security
""")

    # Add energy security guidance if domain knowledge includes it
    if "energy security" in query_lower or "უსაფრთხოება" in query_lower or "independence" in query_lower or "dependence" in query_lower:
        guidance_sections.append("""
CRITICAL: ENERGY SECURITY ANALYSIS RULES:
⚠️ MANDATORY: Thermal generation is import-dependent, NOT local generation!

Key Facts:
- Local/Independent: Hydro, Wind, Solar (no fuel imports)
- Import-Dependent: Thermal (uses imported gas) + Direct electricity import
- Georgia's choice: import electricity OR import gas to generate electricity
- True energy security comes from renewables expansion

When Analyzing Energy Security:
✅ CORRECT: "Winter import dependence includes direct electricity imports AND thermal generation using imported gas"
✅ CORRECT: "Georgia's energy security depends on local renewables (hydro, wind, solar). Thermal generation, while domestic infrastructure, relies on imported gas."
❌ WRONG: "Thermal generation reduces import dependence"
❌ WRONG: "Georgia is self-sufficient when using thermal plants"

Use tech_quantity_view for energy security analysis:
- Sum thermal + import as import-dependent generation
- Sum hydro + wind + solar as local generation
- Calculate shares: local_share = local / (local + import_dependent)
""")

    # General formatting guidelines (always included)
    guidance_sections.append("""
FORMATTING AND LENGTH GUIDELINES:
- When referring to electricity prices or tariffs, always include the correct physical unit (GEL/MWh or USD/MWh) rather than currency only.
- If the question is exploratory or simple (e.g., requesting only a current value, single-month trend, or brief comparison),
  respond in 1–3 clear sentences focusing on the key number or short interpretation.
- If the mode involves correlation, drivers, or in-depth analysis (intent = correlation_analysis, driver_analysis, or trend_analysis),
  write a more detailed summary of about 5–10 sentences following this structure:
  1. Start with the overall yearly trend (using yearly averages).
  2. Present seasonal or period-specific trends if relevant, including CAGRs if available.
  3. If correlation results are provided, discuss primary drivers from domain knowledge.
  4. For GEL vs USD comparisons, explain divergence through exchange rate.
  5. Conclude with a concise analytical insight linking findings to domain knowledge drivers.
- When summarizing, combine numeric findings (averages, CAGRs, correlations) with short explanatory sentences so that the reasoning reads smoothly.
""")

    # Assemble final prompt
    guidance = "\n".join(guidance_sections)

    # Log which guidance sections are included
    guidance_types = []
    if "STAY FOCUSED" in guidance:
        guidance_types.append("focus_rules")
    if "CRITICAL ANALYSIS GUIDELINES for balancing" in guidance:
        guidance_types.append("balancing")
    if "TARIFF ANALYSIS" in guidance:
        guidance_types.append("tariff")
    if "CPI ANALYSIS" in guidance:
        guidance_types.append("cpi")
    if "GENERATION ANALYSIS" in guidance:
        guidance_types.append("generation")
    log.info(f"💬 Answer guidance: focus={query_focus}, sections={guidance_types}")

    prompt = f"""
User question:
{user_query}

Data preview:
{data_preview}

Statistics:
{stats_hint}

Domain knowledge:
{domain_json}

{guidance}
"""

    llm_start = time.time()
    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        out = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        metrics.log_llm_call(time.time() - llm_start)
    except Exception as e:
        log.warning(f"Summarize failed with Gemini, fallback: {e}")
        llm = make_openai()
        out = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        metrics.log_llm_call(time.time() - llm_start)

    # Phase 1 Optimization: Cache the response for future identical requests
    llm_cache.set(cache_input, out)

    return out
