"""Centralized constants for metric computation and share metadata.

Moved here from ``agent.analyzer`` so they can be imported without pulling
in the entire analysis pipeline.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Balancing share metadata – labels, cost buckets, USD linkage
# ---------------------------------------------------------------------------

BALANCING_SHARE_METADATA: dict[str, dict[str, Any]] = {
    "share_regulated_hpp": {"label": "regulated HPP", "cost": "cheap", "usd_linked": False},
    "share_deregulated_hydro": {"label": "deregulated hydro", "cost": "cheap", "usd_linked": False},
    "share_renewable_ppa": {"label": "renewable PPA", "cost": "moderate", "usd_linked": True},
    "share_cfd_scheme": {"label": "CfD scheme", "cost": "moderate", "usd_linked": True},
    "share_thermal_ppa": {"label": "thermal PPA", "cost": "expensive", "usd_linked": True},
    "share_import": {"label": "imports", "cost": "expensive", "usd_linked": True},
    "share_regulated_new_tpp": {"label": "new regulated TPP", "cost": "expensive", "usd_linked": True},
    "share_regulated_old_tpp": {"label": "old regulated TPP", "cost": "moderate", "usd_linked": True},
    "share_all_ppa": {"label": "all PPAs", "cost": "expensive", "usd_linked": True},
    "share_all_renewables": {"label": "all renewables", "cost": "mixed", "usd_linked": True},
}


# ---------------------------------------------------------------------------
# Metric value aliases – canonical name → recognized column names
# ---------------------------------------------------------------------------

METRIC_VALUE_ALIASES: dict[str, list[str]] = {
    # Balancing prices
    "p_bal_gel": ["p_bal_gel", "balancing_price_gel"],
    "p_bal_usd": ["p_bal_usd", "balancing_price_usd"],
    # Deregulated prices
    "p_dereg_gel": ["p_dereg_gel", "deregulated_price_gel"],
    "p_dereg_usd": ["p_dereg_usd", "deregulated_price_usd"],
    # Guaranteed capacity
    "p_gcap_gel": ["p_gcap_gel", "guaranteed_capacity_gel"],
    "p_gcap_usd": ["p_gcap_usd", "guaranteed_capacity_usd"],
    # Exchange rate
    "xrate": ["xrate", "exchange_rate"],
    # Tariffs
    "tariff_gel": ["tariff_gel", "regulated_tariff_gel"],
    "tariff_usd": ["tariff_usd", "regulated_tariff_usd"],
    # Generation
    "quantity_tech": ["quantity_tech", "generation_quantity"],
    "total_demand": ["total_demand"],
    "total_domestic_generation": ["total_domestic_generation"],
    "local_generation": ["local_generation"],
    "import_dependent_supply": ["import_dependent_supply"],
    "import_dependency_ratio": ["import_dependency_ratio"],
    # CPI
    "cpi": ["cpi", "consumer_price_index"],
}


# ---------------------------------------------------------------------------
# Default derived-metric requests (when LLM provides none)
# ---------------------------------------------------------------------------

DERIVED_METRIC_DEFAULTS: list[dict[str, Any]] = [
    # MoM
    {"metric_name": "mom_absolute_change", "metric": "p_bal_gel"},
    {"metric_name": "mom_percent_change", "metric": "p_bal_gel"},
    {"metric_name": "mom_absolute_change", "metric": "xrate"},
    {"metric_name": "mom_percent_change", "metric": "xrate"},
    {"metric_name": "share_delta_mom", "metric": "share_import"},
    {"metric_name": "share_delta_mom", "metric": "share_thermal_ppa"},
    {"metric_name": "share_delta_mom", "metric": "share_renewable_ppa"},
    # YoY — seasonal context for balancing price analysis
    {"metric_name": "yoy_absolute_change", "metric": "p_bal_gel"},
    {"metric_name": "yoy_percent_change", "metric": "p_bal_gel"},
    {"metric_name": "yoy_absolute_change", "metric": "p_bal_usd"},
    {"metric_name": "yoy_percent_change", "metric": "p_bal_usd"},
    {"metric_name": "yoy_absolute_change", "metric": "xrate"},
    {"metric_name": "yoy_percent_change", "metric": "xrate"},
]


# ---------------------------------------------------------------------------
# Seasonal month definitions
# ---------------------------------------------------------------------------

SUMMER_MONTHS = (4, 5, 6, 7)
WINTER_MONTHS = (1, 2, 3, 10, 11, 12)


# ---------------------------------------------------------------------------
# Semantic tool names → DataFrame column candidates
# ---------------------------------------------------------------------------
# The analyzer LLM emits tool-vocabulary metric names (e.g. "balancing")
# in DerivedMetricRequest.metric, but DataFrames use column names
# (e.g. "p_bal_gel").  This map bridges the two vocabularies so the
# metric registry can resolve either form.
#
# Mirrors the mapping in agent/tools/price_tools.py:_resolve_columns().

SEMANTIC_TO_COLUMNS: dict[str, list[str]] = {
    "balancing": ["p_bal_gel", "p_bal_usd"],
    "deregulated": ["p_dereg_gel", "p_dereg_usd"],
    "guaranteed_capacity": ["p_gcap_gel", "p_gcap_usd"],
    "exchange_rate": ["xrate"],
    "demand": ["total_demand"],
    "consumption": ["total_demand"],
    "generation": ["total_domestic_generation", "local_generation"],
    "local_generation": ["local_generation", "total_domestic_generation"],
    "import_dependency": ["import_dependency_ratio", "import_dependent_supply", "quantity_import"],
    "import dependence": ["import_dependency_ratio", "import_dependent_supply", "quantity_import"],
    "energy_security": ["import_dependency_ratio", "local_generation", "total_domestic_generation"],
    "energy security": ["import_dependency_ratio", "local_generation", "total_domestic_generation"],
    "self-sufficiency": ["local_generation", "total_domestic_generation", "import_dependency_ratio"],
    # Share-based metrics for correlation resolution
    "hydro": ["share_all_hydro", "share_deregulated_hydro", "share_regulated_hpp"],
    "import": ["share_import"],
    "renewable": ["share_all_renewables", "share_renewable_ppa"],
    "thermal": ["share_regulated_old_tpp", "share_regulated_new_tpp"],
    "ppa": ["share_all_ppa", "share_renewable_ppa", "share_thermal_ppa"],
    "cfd": ["share_cfd_scheme"],
}
