"""Compact runtime catalogs for the question-analyzer stage."""

from __future__ import annotations

from typing import Any, Dict, List


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
        "use_for": "Requests for data series, tables, or descriptive retrieval without explanation.",
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
        "use_for": "Questions about projection, forecast, future trend, or trendline extension.",
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
        "concepts": ["GENEX", "ESCO", "GSE", "participants", "market roles", "balancing market"],
        "use_for": "Market participants, institutions, and market structure questions.",
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


QUESTION_ANALYSIS_DERIVED_METRIC_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "mom_absolute_change",
        "use_for": "Month-over-month absolute change for a metric when the question asks why a monthly value changed.",
        "examples": ["p_bal_gel", "xrate"],
    },
    {
        "name": "mom_percent_change",
        "use_for": "Month-over-month percentage change for a metric when relative movement matters.",
        "examples": ["p_bal_gel", "xrate"],
    },
    {
        "name": "yoy_absolute_change",
        "use_for": "Year-over-year absolute change for the same month or period.",
        "examples": ["p_bal_gel", "xrate"],
    },
    {
        "name": "yoy_percent_change",
        "use_for": "Year-over-year percent change for the same month or period.",
        "examples": ["p_bal_gel", "xrate"],
    },
    {
        "name": "share_delta_mom",
        "use_for": "Month-over-month share change for balancing composition metrics.",
        "examples": ["share_import", "share_thermal_ppa", "share_renewable_ppa"],
    },
    {
        "name": "correlation_to_target",
        "use_for": "Longer-term correlation between a driver and a target metric.",
        "examples": ["xrate -> p_bal_gel", "share_import -> p_bal_gel"],
    },
    {
        "name": "trend_slope",
        "use_for": "Long-term trend slope for a metric over time.",
        "examples": ["p_bal_gel", "xrate"],
    },
    {
        "name": "scenario_scale",
        "use_for": "Hypothetical scaling: 'What if prices were X% higher/lower?' Set scenario_factor to the multiplier (e.g. 1.34 for 34% higher, 0.8 for 20% lower).",
        "examples": ["p_bal_gel * 1.34", "xrate * 0.9"],
    },
    {
        "name": "scenario_offset",
        "use_for": "Hypothetical offset: 'What if prices were X units higher?' Set scenario_factor to the addend (positive or negative).",
        "examples": ["p_bal_usd + 10", "tariff_gel + 5"],
    },
    {
        "name": "scenario_payoff",
        "use_for": "CfD / PPA payoff calculation: '(strike - market_price) * volume per period'. Set scenario_factor to strike price, scenario_volume to MW capacity (default 1.0).",
        "examples": ["(60 - p_bal_usd) * 1.0", "(50 - p_bal_gel) * 2.5"],
    },
]
