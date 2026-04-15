"""Compact runtime catalogs for the question-analyzer stage."""

from __future__ import annotations

from typing import Any, Dict, List


# Answer-kind guide teaches the analyzer how to classify the expected answer shape.
QUESTION_ANALYSIS_ANSWER_KIND_GUIDE: List[Dict[str, str]] = [
    {
        "name": "scalar",
        "use_for": "Single numeric value or fact for a specific period/entity. Examples: 'What was the price in March?', 'What is the current exchange rate?'",
        "render_style_hint": "deterministic",
    },
    {
        "name": "list",
        "use_for": "Enumeration of entities, categories, or items. Examples: 'Which plants are regulated?', 'List all tariff entities', 'What types of generation exist?'",
        "render_style_hint": "deterministic",
    },
    {
        "name": "timeseries",
        "use_for": "Period-indexed historical data series or table, including historical trend summaries. Examples: 'Show monthly prices for 2025', 'Show the historical trend of balancing prices', 'Generation mix over the last year', 'Tariff history for Enguri'",
        "render_style_hint": "deterministic",
    },
    {
        "name": "comparison",
        "use_for": "Side-by-side comparison of periods, entities, or metrics. Examples: 'Compare Jan vs Feb prices', 'Enguri vs Gardabani tariffs', 'Price in GEL vs USD'",
        "render_style_hint": "deterministic",
    },
    {
        "name": "explanation",
        "use_for": "Why/how questions requiring causal reasoning or driver analysis. Examples: 'Why did prices rise in March?', 'What drives balancing price changes?', 'Explain the price variation'",
        "render_style_hint": "narrative",
    },
    {
        "name": "forecast",
        "use_for": "Explicit forward-looking projection, future estimate, or trendline extension beyond observed data. Examples: 'Forecast prices for next quarter', 'Project generation for 2026', 'Extend the trendline to 2030'. Historical trend summaries stay timeseries, not forecast.",
        "render_style_hint": "deterministic",
    },
    {
        "name": "scenario",
        "use_for": "Hypothetical what-if calculations, CfD/PPA payoffs. Examples: 'What if prices were 20% higher?', 'Calculate CfD payoff at 60 USD strike'",
        "render_style_hint": "deterministic",
    },
    {
        "name": "knowledge",
        "use_for": "Conceptual, definitional, or regulatory procedure questions. Examples: 'What is balancing price?', 'How does GNERC regulate tariffs?', 'Describe the market structure'",
        "render_style_hint": "narrative",
    },
    {
        "name": "clarify",
        "use_for": "The question is too ambiguous or underspecified to answer. The system should ask for clarification.",
        "render_style_hint": "narrative",
    },
]


# Filter guidance teaches the analyzer when to emit value-based filters.
QUESTION_ANALYSIS_FILTER_GUIDE: List[Dict[str, str]] = [
    {
        "pattern": "threshold",
        "use_for": "Questions with numeric thresholds: 'months where price exceeded 15', 'entities with tariff above 10'",
        "example_filter": '{"metric": "balancing", "operator": "gt", "value": 15, "unit": "tetri"}',
    },
    {
        "pattern": "change_filter",
        "use_for": "Questions filtering by change magnitude: 'months where price increased by more than 5%'",
        "example_filter": '{"metric": "balancing", "operator": "gt", "value": 5, "unit": "percent"}',
    },
]


# Query-type guidance teaches the analyzer how to classify the user request.
QUESTION_ANALYSIS_QUERY_TYPE_GUIDE: List[Dict[str, str]] = [
    {
        "name": "conceptual_definition",
        "use_for": "Definition or meaning questions about terms, entities, or market concepts.",
    },
    {
        "name": "regulatory_procedure",
        "use_for": "Questions about regulatory processes, registration steps, eligibility rules, required documents, participation conditions, or deadlines. Not for simple term definitions.",
    },
    {
        "name": "factual_lookup",
        "use_for": "Single fact or direct value lookup for a known entity, metric, or period.",
    },
    {
        "name": "data_retrieval",
        "use_for": "Requests for historical data series, trend summaries, tables, quantitative value lookups ('how much', 'what was the total'), or descriptive retrieval without explanation.",
    },
    {
        "name": "data_explanation",
        "use_for": "Why/how questions about observed change, variation, or drivers in data.",
    },
    {
        "name": "comparison",
        "use_for": "Queries comparing entities, periods, or metrics.",
    },
    {
        "name": "forecast",
        "use_for": "Questions about explicit forward-looking projection, forecast, future estimate, or trendline extension beyond observed data. Historical trend summaries are data_retrieval, not forecast.",
    },
    {
        "name": "ambiguous",
        "use_for": "The query is too unclear or underspecified to route confidently.",
    },
    {
        "name": "unsupported",
        "use_for": "The query is outside the supported scope of the system.",
    },
]


# Topic catalog tells the analyzer which knowledge domains are available.
QUESTION_ANALYSIS_TOPIC_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "general_definitions",
        "concepts": ["definition", "terminology", "general electricity market concepts"],
        "use_for": "General definitions and conceptual electricity-market explanations.",
    },
    {
        "name": "balancing_price",
        "concepts": ["balancing price", "p_bal", "price drivers", "shares", "xrate", "variation"],
        "use_for": "Balancing price formation, variation, drivers, and seasonal patterns.",
    },
    {
        "name": "market_structure",
        "concepts": ["GENEX", "ESCO", "GSE", "participants", "market roles", "balancing market",
                      "transitory period", "market design", "market model", "target model"],
        "use_for": "Market participants, institutions, market structure, and market model design questions.",
    },
    {
        "name": "tariffs",
        "concepts": ["regulated tariff", "GNERC", "Enguri", "Gardabani", "cost-plus"],
        "use_for": "Tariff entities, tariff logic, and regulated pricing questions.",
    },
    {
        "name": "cfd_ppa",
        "concepts": ["CfD", "PPA", "support scheme", "strike price"],
        "use_for": "Support schemes, PPA/CfD concepts, and scheme-specific explanations.",
    },
    {
        "name": "pso_trading",
        "concepts": ["PSO", "public service obligation", "Telmico", "EP Georgia Supply",
                      "procurement", "universal service", "public service", "cascade distribution"],
        "use_for": "PSO supplier procurement structures, Telmico and EP Georgia trading results, and Enguri/Vardnili cascade supply distribution.",
    },
    {
        "name": "currency_influence",
        "concepts": ["exchange rate", "GEL/USD", "xrate", "USD-linked costs"],
        "use_for": "Exchange-rate impact and currency-linked price pressure.",
    },
    {
        "name": "seasonal_patterns",
        "concepts": ["seasonality", "summer", "winter", "seasonal trend"],
        "use_for": "Seasonal market behavior and seasonal trend questions.",
    },
    {
        "name": "generation_mix",
        "concepts": ["generation", "technology mix", "hydro", "thermal", "wind", "solar"],
        "use_for": "Generation composition and generation quantity questions.",
    },
    {
        "name": "sql_examples",
        "concepts": ["query patterns", "SQL examples", "analysis templates"],
        "use_for": "Planning support for SQL-oriented data queries.",
    },
]


# Tool catalog describes when each deterministic tool should or should not be used.
QUESTION_ANALYSIS_TOOL_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "get_prices",
        "concepts": [
            "balancing price",
            "deregulated price",
            "guaranteed capacity price",
            "exchange rate",
            "GEL",
            "USD",
            "trend",
        ],
        "use_for": "Price or exchange-rate retrieval over time or for a stated period.",
        "avoid_for": "Conceptual definitions, tariff questions, and share-only composition queries.",
        "main_params": ["metric", "currency", "granularity", "start_date", "end_date"],
        "metric_values": ["balancing", "deregulated", "guaranteed_capacity", "exchange_rate"],
        "metric_hint_rules": [
            "Use tool enums, not DB columns or chart aliases.",
            "balancing price in GEL/USD -> metric=balancing with currency=gel/usd",
            "exchange rate or xrate -> metric=exchange_rate",
            "Never emit p_bal_gel, p_bal_usd, p_dereg_gel, p_gcap_gel, or balancing_price_gel as metric values",
        ],
        "combined_with": [
            {"tool": "get_balancing_composition", "when": "explaining price drivers or composition effects on price"},
            {"tool": "get_tariffs", "when": "analyzing regulated cost impact on balancing price"},
        ],
    },
    {
        "name": "get_tariffs",
        "concepts": ["regulated tariffs", "GNERC", "Enguri", "Gardabani", "cost-plus"],
        "use_for": "Tariff lookups and tariff comparisons.",
        "avoid_for": "Balancing price questions, conceptual definitions, and generation mix questions.",
        "main_params": ["entities", "currency", "start_date", "end_date"],
        "combined_with": [
            {"tool": "get_prices", "when": "comparing tariffs against market prices"},
        ],
    },
    {
        "name": "get_generation_mix",
        "concepts": ["generation", "technology mix", "hydro", "thermal", "wind", "solar", "quantity", "share"],
        "use_for": "Generation mix or quantity by technology or type.",
        "avoid_for": "Tariffs, balancing price, and conceptual definitions.",
        "main_params": ["types", "mode", "granularity", "start_date", "end_date"],
        "combined_with": [
            {"tool": "get_prices", "when": "correlating generation patterns with price movements"},
        ],
    },
    {
        "name": "get_balancing_composition",
        "concepts": ["balancing shares", "composition", "imports", "PPA share", "hydro share"],
        "use_for": "Balancing market composition and share questions.",
        "avoid_for": "Price-only questions, conceptual definitions, and tariffs.",
        "main_params": ["entities", "start_date", "end_date"],
        "combined_with": [
            {"tool": "get_prices", "when": "understanding how composition shifts affect prices"},
        ],
    },
]


# Chart policy keeps chart suggestions lightweight and query-type aware.
QUESTION_ANALYSIS_CHART_POLICY: List[Dict[str, str]] = [
    {
        "case": "definition_or_conceptual",
        "hint": "Usually no chart.",
    },
    {
        "case": "single_period_explanation",
        "hint": "Usually no chart unless the user explicitly asks.",
    },
    {
        "case": "trend_or_time_series",
        "hint": "Usually recommend a line chart.",
    },
    {
        "case": "share_or_composition",
        "hint": "Usually recommend stacked or composition-style charts.",
    },
]


# Derived-metric catalog documents the analytical calculations the runtime can materialize.
QUESTION_ANALYSIS_DERIVED_METRIC_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "mom_absolute_change",
        "use_for": "Month-over-month absolute change for a metric when the question asks why a monthly value changed.",
        "examples": ["balancing", "exchange_rate"],
        "metric_note": "Use tool-vocabulary names (balancing, deregulated, exchange_rate) for price metrics. Use column names (share_import, share_thermal_ppa) for share metrics.",
    },
    {
        "name": "mom_percent_change",
        "use_for": "Month-over-month percentage change for a metric when relative movement matters.",
        "examples": ["balancing", "exchange_rate"],
    },
    {
        "name": "yoy_absolute_change",
        "use_for": "Year-over-year absolute change for the same month or period.",
        "examples": ["balancing", "exchange_rate"],
    },
    {
        "name": "yoy_percent_change",
        "use_for": "Year-over-year percent change for the same month or period.",
        "examples": ["balancing", "exchange_rate"],
    },
    {
        "name": "share_delta_mom",
        "use_for": "Month-over-month share change for balancing composition metrics.",
        "examples": ["share_import", "share_thermal_ppa", "share_renewable_ppa"],
    },
    {
        "name": "correlation_to_target",
        "use_for": "Longer-term correlation between a driver and a target metric.",
        "examples": ["exchange_rate -> balancing", "share_import -> balancing"],
    },
    {
        "name": "trend_slope",
        "use_for": "Long-term trend slope for a metric over time. For seasonal comparisons, emit two requests with season='summer' and season='winter'.",
        "examples": ["balancing", "exchange_rate"],
    },
    {
        "name": "scenario_scale",
        "use_for": "Hypothetical scaling: 'What if prices were X% higher/lower?' Set scenario_factor to the multiplier (e.g. 1.34 for 34% higher, 0.8 for 20% lower).",
        "examples": ["balancing * 1.34", "exchange_rate * 0.9"],
    },
    {
        "name": "scenario_offset",
        "use_for": "Hypothetical offset: 'What if prices were X units higher?' Set scenario_factor to the addend (positive or negative).",
        "examples": ["balancing + 10", "tariff_gel + 5"],
    },
    {
        "name": "scenario_payoff",
        "use_for": "CfD / PPA payoff calculation: '(strike - market_price) * volume per period'. Set scenario_factor to strike price, scenario_volume to MW capacity (default 1.0).",
        "examples": ["(60 - balancing) * 1.0", "(50 - balancing) * 2.5"],
    },
]
