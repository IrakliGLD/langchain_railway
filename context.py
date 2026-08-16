# === context.py v2.1 ===
# Updated for main.py v18.5: materialized-view schema, OpenAI context alignment, demand/supply groups.

import re

# --- Column label mapping ---
COLUMN_LABELS = {
    # shared
    "date": "Date",


    # entities_mv
    "entity": "Entity Name",
    "entity_normalized": "Standardized Entity ID",
    "type": "Entity Type",
    "ownership": "Ownership",
    "source": "Source",


    # price_with_usd
    "p_dereg_gel": "Deregulated Price (GEL/MWh)",
    "p_bal_gel": "Balancing electricity price (GEL/MWh)",
    "p_gcap_gel": "Guaranteed Capacity Fee (GEL/MWh)",
    "xrate": "Exchange Rate (GEL/USD)",
    "p_dereg_usd": "Deregulated Price (USD/MWh)",
    "p_bal_usd": "Balancing electricity price (USD/MWh)",
    "p_gcap_usd": "Guaranteed Capacity Fee (USD/MWh)",

    # tariff_with_usd
    "tariff_gel": "Regulated Tariff (GEL/MWh)",
    "tariff_usd": "Regulated Tariff (USD/MWh)",

    # tech_quantity_view
    "type_tech": "Technology Type",
    "quantity_tech": "Quantity (thousand MWh)",

    # trade_derived_entities
    "segment": "Market Segment",
    "quantity": "Trade Volume (thousand MWh)",

    # monthly_cpi_mv
    "cpi_type": "CPI Category",
    "cpi": "Consumer Price Index",

    # mv_balancing_trade_with_tariff
    "month": "Month",
    "entity_code": "Entity Code",
    "balancing_quantity": "Balancing Quantity (thousand MWh)",

    # energy_balance_long_mv
    "year": "Year",
    "sector": "Energy Balance Sector",
    "energy_source": "Energy Source",
    "volume_tj": "Energy Volume (TJ)",

    # by_capacity / by_commissioning / capacity_factor
    "facility_count": "Facility Count",
    "technology": "Technology",
    "capacity_category": "Installed Capacity Band",
    "capacity_category_order": "Capacity Band Sort Order",
    "generation_mwh": "Generation (MWh)",
    "installed_capacity_mw": "Installed Capacity (MW)",
    "hours_in_month": "Hours in Month",
    "capacity_factor": "Capacity Factor (ratio 0-1)",
    "capacity_factor_percent": "Capacity Factor (%)",

    # ownership_concentration
    "total_generation": "Total Generation (thousand MWh)",
    "owner_count": "Number of Owners",
    "hhi": "Herfindahl-Hirschman Index",
    "top1_share": "Top 1 Owner Share",
    "top3_share": "Top 3 Owner Share",
    "top5_share": "Top 5 Owner Share",

    # demand_tariff_mv
    "company": "Tariff Company",
    "activity": "Tariff Activity",
    "volate": "Voltage Level",
    "level_1_cat": "Consumer Class",
    "level_2_cat": "Consumer Sub-Class",
    "value": "Tariff Component (GEL/kWh)",

    # get_end_user_prices output frame. Each name carries its unit: these are
    # GEL/kWh while wholesale prices are GEL/MWh, and a bare "supply" reads as
    # a volume to is_intensive_metric, licensing a summed per-kWh tariff.
    "supplier": "Supply Company Code",
    "supply_company": "Supply Company",
    "series_label": "Company and Category",
    # Distinct labels: these are two different columns (the machine key and
    # its human string), and the label -> identifier inverse must round-trip.
    "category": "End-User Category Code",
    "category_label": "End-User Category",
    "distribution_tariff_gel_kwh": "Distribution tariff (GEL/kWh)",
    "supply_tariff_gel_kwh": "Supply tariff, incl. guaranteed capacity fee (GEL/kWh)",
    "transmission_tariff_gel_kwh": "Transmission tariff (GEL/kWh)",
    "final_price_net_gel_kwh": "Final end-user price, net of VAT (GEL/kWh)",
    "vat_gel_kwh": "VAT at 18% (GEL/kWh)",
    "total_gross_gel_kwh": "Final end-user price, incl. VAT (GEL/kWh)",
    "wholesale_benchmark_gel_kwh": (
        "Wholesale benchmark: balancing + guaranteed capacity + ESCO fee (GEL/kWh)"
    ),
    "supply_vs_wholesale_spread_gel_kwh": "Supply component vs wholesale, spread (GEL/kWh)",
}

# --- Columns excluded from scrub_schema_mentions ---
# These keys MUST stay in COLUMN_LABELS: readiness and display read that dict,
# and tests/test_context.py requires every DB_SCHEMA_DICT column to have a
# label.  They MUST NOT be substituted into narrative text, because each is an
# ordinary English word and the substitution is a case-insensitive
# ``\b{key}\b`` replacement over the LLM's prose.
#
# This is the same failure already documented for VALUE_LABELS below: a bare
# common-English key mangles ordinary sentences.  Concretely, without this set
# "the value of imports rose" becomes "the Tariff Component (GEL/kWh) of
# imports rose", and "wind technology" becomes "wind Technology".
#
# Identifier-shaped keys (snake_case with an underscore, acronyms like HPP,
# hyphenated codes) never occur in natural prose and stay scrubbed.
SCRUB_EXEMPT_COLUMNS = frozenset({
    "value",
    "activity",
    "company",
    "technology",
    "supplier",
    "category",
})

# ----------------------------------------------------------
# DERIVED_LABELS — for LLM-generated / computed columns
# (These do not exist physically in Supabase views)
# ----------------------------------------------------------
DERIVED_LABELS = {
    "share_import": "Share of Imports in Balancing Electricity",
    "share_cfd_scheme": "Share of CfD scheme supported generation in Balancing Electricity",
    "share_deregulated_ren": "Share of Deregulated Renewable",
    "share_regulated_hpp": "Share of Regulated HPPs",
    "share_regulated_new_tpp": "Share of Regulated New TPPs",
    "share_regulated_old_tpp": "Share of Regulated Old TPPs",
    "share_total_hpp": "Share of Total HPP Output",
    "share_renewable_ppa": "Share of Renewable PPAs",
    "share_thermal_ppa": "Share of Thermal PPAs",
    "share_all_ppa": "Share of All PPAs",
    "share_all_renewables": "Share of All Renewable Sources",
    "enguri_tariff_gel": "Enguri HPP Tariff (GEL/MWh)",
    "enguri_tariff_usd": "Enguri HPP Tariff (USD/MWh)",
    "gardabani_tpp_tariff_gel": "Gardabani TPP Tariff (GEL/MWh)",
    "gardabani_tpp_tariff_usd": "Gardabani TPP Tariff (USD/MWh)",
    "grouped_old_tpp_tariff_gel": "Old Thermal Power Plants Tariff (GEL/MWh)",
    "grouped_old_tpp_tariff_usd": "Old Thermal Power Plants Tariff (USD/MWh)",
    "regulated_hpp_tariff_gel": "Regulated HPP Tariff (GEL/MWh)",
    "regulated_hpp_tariff_usd": "Regulated HPP Tariff (USD/MWh)",
    "regulated_new_tpp_tariff_gel": "Regulated New TPP Tariff (GEL/MWh)",
    "regulated_new_tpp_tariff_usd": "Regulated New TPP Tariff (USD/MWh)",
    "regulated_old_tpp_tariff_gel": "Regulated Old TPP Tariff (GEL/MWh)",
    "regulated_old_tpp_tariff_usd": "Regulated Old TPP Tariff (USD/MWh)",
    "weighted_gel": "Weighted-Average Balancing Price (GEL/MWh)",
    "weighted_usd": "Weighted-Average Balancing Price (USD/MWh)",
    "quantity_hydro": "Hydro Generation (thousand MWh)",
    "quantity_thermal": "Thermal Generation (thousand MWh)",
    "quantity_wind": "Wind Generation (thousand MWh)",
    "quantity_solar": "Solar Generation (thousand MWh)",
    "quantity_import": "Direct Electricity Imports (thousand MWh)",
    "quantity_export": "Electricity Exports (thousand MWh)",
    "quantity_losses": "System Losses (thousand MWh)",
    "quantity_abkhazeti": "Abkhazeti Demand (thousand MWh)",
    "quantity_supply-distribution": "Supplier/Distributor Demand (thousand MWh)",
    "quantity_direct customers": "Direct Customer Demand (thousand MWh)",
    "share_hydro": "Share of Hydro Generation",
    "share_thermal": "Share of Thermal Generation",
    "share_wind": "Share of Wind Generation",
    "share_solar": "Share of Solar Generation",
    "share_export": "Share of Electricity Exports",
    "share_losses": "Share of System Losses",
    "share_abkhazeti": "Share of Abkhazeti Demand",
    "share_supply-distribution": "Share of Supplier/Distributor Demand",
    "share_direct customers": "Share of Direct Customer Demand",
    "total_demand": "Total Electricity Demand (thousand MWh)",
    "total_domestic_generation": "Total Domestic Generation (thousand MWh)",
    "local_generation": "Local Non-Import-Dependent Generation (thousand MWh)",
    "import_dependent_supply": "Import-Dependent Supply (thousand MWh)",
    "total_supply": "Total Supply Available (thousand MWh)",
    "import_dependency_ratio": "Import Dependency Ratio",
    "season": "Season (Summer/Winter)",
    "period_group": "Period Group (e.g., 2015–2020 vs 2021–2025)",
}

# --- Table label mapping ---
VIEW_LABELS = {
    "entities_mv": "Power Sector Entities",
    "price_with_usd": "Electricity Market Prices (USD)",
    "tariff_with_usd": "Regulated Tariffs (USD)",
    "tech_quantity_view": "Generation & Demand Quantities",
    "trade_derived_entities": "Electricity Trade",
    "energy_balance_long_mv": "Annual Energy Balance",
    "monthly_cpi_mv": "Monthly Consumer Price Index",
    "dates_mv": "Date Reference",
    "mv_balancing_trade_with_tariff": "Balancing Tariffs by Entity",
    "trade_by_ownership": "Generation by Owner Group",
    "ownership_concentration": "Generation Ownership Concentration",
    "by_capacity": "Generation by Installed Capacity Band",
    "by_commissioning": "Generation by Commissioning Cohort",
    "capacity_factor": "Capacity Factor by Technology and Band",
    "demand_tariff_mv": "Regulated End-User Tariff Components",
}

# --- Demand/Supply classification for type_tech ---
# Note: Keys match actual database values (with dashes/spaces as stored in DB)
TECH_TYPE_GROUPS = {
    "demand": {
        "abkhazeti": "Abkhazeti",
        "supply-distribution": "Supplier/Distributor",
        "direct customers": "Direct Consumers",
        "losses": "Losses",
        "export": "Export",
    },
    "supply": {
        "hydro": "Hydro Generation",
        "thermal": "Thermal Generation",
        "wind": "Wind Generation",
        "import": "Import",
        "solar": "Solar Generation",
    },
}

# --- Value label mapping (scrub_schema_mentions ONLY) ---
# INVARIANT: every key here is replaced by a case-insensitive ``\b{key}\b``
# substitution on LLM narrative output.  Therefore a key MUST be a
# schema/identifier-shaped token (snake_case column value, acronym, hyphenated
# code) that a user should never see raw — NEVER an ordinary English word.
#
# Bare common-English words are intentionally EXCLUDED (see the balancing
# precedent + tests/test_context.py). Two failure modes they cause:
#   1. Redundant doubling when the label appends a category noun the narrative
#      already wrote: VALUE_LABELS["hydro"]="Hydro Generation" turned the
#      LLM's "hydro generation" into "Hydro Generation generation"
#      (2026-07-08 production report), and "transit"→"Transit Flows" would
#      double "transit flows".
#   2. Mid-sentence mangling: "balancing"→"Balancing Electricity" produced
#      "the Balancing Electricity price".
# Excluded bare words (kept in TECH_TYPE_GROUPS for classification, which uses
# the KEYS only — see agent/tools/generation_tools.py): hydro, thermal, wind,
# solar, import, export, transit, losses, direct customers, balancing.
# These read perfectly well as-is, so exclusion is loss-free for scrubbing.
VALUE_LABELS = {
    # Proper noun / hyphenated code (not ordinary prose) — safe to relabel.
    "abkhazeti": TECH_TYPE_GROUPS["demand"]["abkhazeti"],
    "supply-distribution": TECH_TYPE_GROUPS["demand"]["supply-distribution"],
    # Acronyms.
    "HPP": "Hydropower Plant",
    "TPP": "Thermal Power Plant",
    # snake_case / code-shaped composition tokens the LLM must not leak raw.
    "bilateral_exchange": "Bilateral Contracts & Exchange",
    "renewable_ppa": "Renewable PPA",
    "thermal_ppa": "Thermal PPA",
    "deregulated_ren": "Deregulated Renewable",
    "regulated_hpp": "Regulated HPP",
    "regulated_new_tpp": "Regulated new TPP",
    "regulated_old_tpp": "Regulated old TPP",
    "CfD_scheme": "CfD Scheme",
}

# --- Structured Schema Dict ---
DB_SCHEMA_DICT = {
    "views": {
        "entities_mv": {
            "columns": ["entity", "entity_normalized", "type", "ownership", "source"],
            "desc": "Power Sector Entities",
        },

        "price_with_usd": {
            "columns": ["date", "p_dereg_gel", "p_bal_gel", "p_gcap_gel", "xrate", "p_dereg_usd", "p_bal_usd", "p_gcap_usd"],
            "desc": "Electricity Market Prices (GEL and USD)",
        },
        "tariff_with_usd": {
            "columns": ["date", "entity", "tariff_gel", "tariff_usd"],
            "desc": "Regulated Tariffs (GEL and USD)",
        },
        "tech_quantity_view": {
            "columns": ["date", "type_tech", "quantity_tech"],
            "desc": "Generation & Demand Quantities by Technology Type",
        },
        "trade_derived_entities": {
            "columns": ["date", "entity", "segment", "quantity"],
            "desc": "Electricity Trade Volumes (Derived)",
        },
        "energy_balance_long_mv": {
            "columns": ["year", "sector", "energy_source", "volume_tj"],
            "desc": "Annual Energy Balance",
        },
        "monthly_cpi_mv": {
            "columns": ["date", "cpi_type", "cpi"],
            "desc": "Monthly Consumer Price Index",
        },
        "dates_mv": {
            "columns": ["date"],
            "desc": "Date Reference (Utility View)",
        },
        "mv_balancing_trade_with_tariff": {
            "columns": ["month", "entity", "entity_code", "tariff_gel", "balancing_quantity"],
            "desc": "Balancing Market Tariffs by Entity (monthly regulated tariff with balancing quantity per entity)",
        },
        "trade_by_ownership": {
            "columns": ["date", "ownership", "quantity"],
            "desc": "Monthly Generation by Owner Group",
        },
        "ownership_concentration": {
            "columns": [
                "date", "segment", "total_generation", "owner_count",
                "hhi", "top1_share", "top3_share", "top5_share",
            ],
            "desc": "Monthly Generation Ownership Concentration (HHI and top-N shares)",
        },
        "by_capacity": {
            "columns": ["date", "entity", "segment", "quantity", "facility_count"],
            "desc": "Monthly Generation and Facility Count by Installed Capacity Band",
        },
        "by_commissioning": {
            "columns": ["date", "entity", "segment", "quantity"],
            "desc": "Monthly Generation by Commissioning Cohort",
        },
        "capacity_factor": {
            "columns": [
                "date", "technology", "capacity_category", "capacity_category_order",
                "segment", "facility_count", "generation_mwh", "installed_capacity_mw",
                "hours_in_month", "capacity_factor", "capacity_factor_percent",
            ],
            "desc": "Monthly Capacity Factor by Technology and Installed Capacity Band",
        },
        # ``demand_tariff_id`` exists on the view but is deliberately omitted: it
        # is NULL on every calculated row, so it is useless as a join key.
        # REQUIRED_SCHEMA_COLUMNS checks required-subset-of-reflected, so leaving
        # a real column out is safe.
        "demand_tariff_mv": {
            "columns": [
                "date", "company", "activity", "volate",
                "level_1_cat", "level_2_cat", "value",
            ],
            "desc": "Regulated End-User Tariff Components (distribution, supply, transmission; GEL/kWh)",
        },
    },
    "rules": {

        "usd_rule": "USD values = GEL / xrate (from price_with_usd joined by date).",
        "granularity": "Monthly data for all except yearly energy_balance_long_mv.",
        "temporal_scope": "2015–present; use full range.",
    },
}

# --- Reinforced Schema Text for LLM Context ---
DB_SCHEMA_DOC = """
### Key Database Rules and Conventions (Materialized Views)

**Available Views:**
- entities_mv(entity, entity_normalized, type, ownership, source)
- price_with_usd(date, p_dereg_gel, p_bal_gel, p_gcap_gel, xrate, p_dereg_usd, p_bal_usd, p_gcap_usd)
- tariff_with_usd(date, entity, tariff_gel, tariff_usd)
- mv_balancing_trade_with_tariff(month, entity, entity_code, tariff_gel, balancing_quantity)
- tech_quantity_view(date, type_tech, quantity_tech)
- trade_derived_entities(date, entity, segment, quantity)
- energy_balance_long_mv(year, sector, energy_source, volume_tj)
- trade_by_ownership(date, ownership, quantity)
- ownership_concentration(date, segment, total_generation, owner_count, hhi, top1_share, top3_share, top5_share)
- by_capacity(date, entity, segment, quantity, facility_count)
- by_commissioning(date, entity, segment, quantity)
- capacity_factor(date, technology, capacity_category, capacity_category_order, segment, facility_count, generation_mwh, installed_capacity_mw, hours_in_month, capacity_factor, capacity_factor_percent)
- demand_tariff_mv(date, company, activity, volate, level_1_cat, level_2_cat, value)

**CRITICAL: Exact column values (case-sensitive, including spaces/hyphens):**

type_tech values (tech_quantity_view):
- Demand side: 'abkhazeti', 'supply-distribution' (note: hyphen!), 'direct customers' (note: space! - MARKET CATEGORY not industry sector, see DirectCustomers domain knowledge), 'losses', 'export'
- Supply side: 'hydro', 'thermal', 'wind', 'import', 'solar'
- IMPORTANT: Use exact strings with hyphens and spaces as shown above!

segment values (trade_derived_entities):
- For balancing-segment trade, use the canonical normalized segment token 'balancing'
- IMPORTANT: User phrasing like "balancing electricity" refers to electricity traded in the balancing segment
- Recommended filter: WHERE LOWER(REPLACE(segment, ' ', '_')) = 'balancing'

entity values (trade_derived_entities, balancing segment):
- 'import', 'deregulated_ren', 'regulated_hpp', 'regulated_new_tpp',
  'regulated_old_tpp', 'renewable_ppa', 'thermal_ppa', 'CfD_scheme'
- IMPORTANT: Use exact strings as shown above! Note CfD_scheme uses mixed case.

mv_balancing_trade_with_tariff notes:
- 'month' = first day of each month (same granularity as 'date' in other views)
- 'entity' = entity_normalized name (plant-level, e.g. from entities_mv)
- 'entity_code' = raw entity code from trade/tariff tables
- Contains only regulated entities that sold on the balancing segment
- tariff_gel = regulated tariff for that entity; balancing_quantity = energy sold on balancing
- JOIN to other views: mv_balancing_trade_with_tariff.month = price_with_usd.date (cross-column join)

**Plant-fleet views (by_capacity, by_commissioning, capacity_factor, ownership_concentration, trade_by_ownership):**

- by_capacity.entity AND capacity_factor.capacity_category share ONE 8-band vocabulary (MW), in
  this order: '<=5', '6-10', '11-20', '21-50', '51-100', '101-200', '201-500', 'more than 500'.
  These are TEXT: ORDER BY the band column sorts 101-200 before 11-20. Use
  capacity_factor.capacity_category_order; by_capacity has no order column, so sort by the list.
- by_commissioning.entity (commissioning cohort): '<=1990', '1991-2000', '2001-2010', '2011-2020', 'after 2020'
- capacity_factor.technology: 'hpp', 'tpp', 'wpp', 'solar'. technology × band is SPARSE.
- trade_by_ownership.ownership, exact case ('GIG' uppercase, rest lowercase): 'energo-pro group',
  'georgian water and power jcs', 'GIG', 'inter-rao', 'other', 'state', 'vartsikhe 2005 jsc'
- SCALES (the main failure mode here):
  * capacity_factor.generation_mwh is plain MWh; `quantity` and total_generation are THOUSAND
    MWh — exactly 1000x. Never add or compare them unconverted.
  * `capacity_factor` is a ratio 0..1, `capacity_factor_percent` is that ×100. Pick one.
    Never multiply capacity_factor_percent by 100.
  * `hhi` is 0-10000; `top1_share`/`top3_share`/`top5_share` are ratios 0..1.
- capacity_factor is precomputed (generation_mwh / (installed_capacity_mw * hours_in_month)):
  read the column, never recompute.
- by_capacity measures GENERATION per capacity band, not installed capacity — the band label is
  the MW range. `installed_capacity_mw` exists only in capacity_factor.
- by_capacity and by_commissioning both partition the same monthly total, which equals
  ownership_concentration.total_generation.
- trade_by_ownership is TRADE, not generation: its monthly total does NOT equal
  total_generation. Never present one as a share of the other.
- `segment` in these four views holds exactly ONE value, 'total' (trade_by_ownership has no
  segment column at all). Do NOT reuse the trade_derived_entities 'balancing' filter here — it
  matches nothing and returns an empty result.

**demand_tariff_mv — regulated END-USER tariff components (GEL/kWh, NOT GEL/MWh):**

- company: 'telasi', 'epg' (distribution) | 'telmico', 'eps' (supply) | 'gse' (transmission)
- activity: 'distribution', 'universal', 'public', 'transmission', 'final_price', 'solr'
- volate (voltage level, stored verbatim): '', '220/380', '3.3-6-10', '35-110'
- level_1_cat: '', 'com' (commercial), 'hh' (household)
- level_2_cat: '', 'cat1' (<=101 kWh), 'cat2' (101-301 kWh), 'cat3' (>301 kWh), 'other', 'small'
- Blank dimensions are EMPTY STRINGS (''), never NULL. Filter with = '' — IS NULL matches nothing.
- Report demand_tariff_mv values in GEL/kWh as stored. Do NOT convert to GEL/MWh: the converted
  number exists in no row and the grounding gate will strip it from the answer.
- value and final_price are NET of VAT. VAT is 18% and is levied on top, so a consumer pays
  final_price × 1.18. Report the net figure by default and say it is net of VAT; give the gross
  only when the question asks what a consumer actually pays.
- To compare with wholesale, join price_with_usd on date and use
  (p_bal_gel + p_gcap_gel) / 1000 as the GEL/kWh benchmark. The supply tariff already bundles
  the guaranteed capacity fee, so it is added to the WHOLESALE side; compare against net.

The end-user price is the SUM of three components for one (date, supplier, category):
  1. distribution — the supplier's distributor: 'telmico'->'telasi', 'eps'->'epg'
  2. supply       — the supplier itself; activity 'universal' (households / small commercial)
                    or 'public' (public-service commercial)
  3. transmission — company='gse', activity='transmission', volate/level_1_cat/level_2_cat all ''
'final_price' rows are the publisher's own pre-summed total — use them to CROSS-CHECK a computed
sum, not to build one. 'solr' (supplier of last resort) is not part of the end-user price.

USABLE RANGE: rows run to 2030-12-01, but a complete end-user price exists only where final_price
rows exist. Distribution tariffs are published years ahead of the supply and transmission rows they
must combine with, so MAX(date) on this view is misleading. For end-user price questions bound the
window with (SELECT MAX(date) FROM demand_tariff_mv WHERE activity = 'final_price').

**Units & Conversions:**
- Quantities in thousand MWh (multiply ×1000 for MWh)
- *_usd fields = *_gel / xrate
- THREE scales now coexist — never mix them in one expression:
  * thousand MWh — quantity / quantity_tech (trade, tech, and plant-fleet views)
  * MWh and MW   — capacity_factor.generation_mwh, capacity_factor.installed_capacity_mw
  * GEL/kWh      — demand_tariff_mv.value (every other price and tariff is GEL/MWh)
- demand_tariff_mv in USD/kWh = value / xrate (join price_with_usd on date)

**Granularity:**
- Monthly for all except energy_balance_long_mv (yearly)

**Derived Dimensions:**
- Season is not a column in the database but can be computed analytically as:
  CASE WHEN EXTRACT(MONTH FROM date) IN (4,5,6,7) THEN 'Summer' ELSE 'Winter' END AS season
- Use this derived field for seasonal aggregation of prices (AVG) or quantities (SUM).

**Joins:**
- Use date or entity as join keys only.
- Avoid system tables or undefined joins.

**Time Coverage:** 2015–latest month

**CRITICAL – Per-View Data Availability:**
- price_with_usd: complete from 2006 onwards
- tariff_with_usd: complete from 2008 onwards
- tech_quantity_view: complete from 2014 onwards
- trade_derived_entities: complete from **2020** onwards (NO data before 2020)
  * If shares are NULL for a period → data is NOT available, do NOT treat as 0%
  * For balancing composition analysis, always filter: date >= '2020-01-01'
- mv_balancing_trade_with_tariff: complete from **2020** onwards (derived from trade + tariff_gen)
  * NULL for a regulated group in a month means that group had no balancing sales — do NOT treat as 0
- trade_by_ownership, ownership_concentration, by_capacity, by_commissioning, capacity_factor:
  complete from **2020-01** onwards
- demand_tariff_mv: from **2021-07** onwards; complete end-user prices only through the latest
  final_price month (see USABLE RANGE above) — later rows are distribution-only
"""

# --- Join Map ---
DB_JOINS = {
    "price_with_usd": {"join_on": "date", "related_to": ["tariff_with_usd", "tech_quantity_view", "trade_derived_entities"]},
    "tariff_with_usd": {"join_on": ["date", "entity"], "related_to": ["price_with_usd", "trade_derived_entities"]},
    "tech_quantity_view": {"join_on": "date", "related_to": ["price_with_usd", "trade_derived_entities"]},
    "trade_derived_entities": {"join_on": ["date", "entity"], "related_to": ["price_with_usd", "tariff_with_usd"]},
    "entities_mv": {"join_on": "entity", "related_to": ["tariff_with_usd", "trade_derived_entities"]},
    "energy_balance_long_mv": {"join_on": "year", "related_to": []},
    "monthly_cpi_mv": {"join_on": "date", "related_to": ["price_with_usd"]},
    "dates_mv": {"join_on": "date", "related_to": ["price_with_usd", "tariff_with_usd", "tech_quantity_view", "trade_derived_entities", "monthly_cpi_mv"]},
    "mv_balancing_trade_with_tariff": {"join_on": "month", "related_to": ["price_with_usd"], "join_note": "month = price_with_usd.date (different column names)"},
    "trade_by_ownership": {"join_on": "date", "related_to": ["price_with_usd", "ownership_concentration"]},
    "ownership_concentration": {"join_on": "date", "related_to": ["price_with_usd", "trade_by_ownership"]},
    "by_capacity": {"join_on": "date", "related_to": ["price_with_usd", "capacity_factor"], "join_note": "by_capacity.entity shares the 8-band vocabulary with capacity_factor.capacity_category"},
    "by_commissioning": {"join_on": "date", "related_to": ["price_with_usd"]},
    "capacity_factor": {"join_on": "date", "related_to": ["price_with_usd", "by_capacity"], "join_note": "capacity_factor.capacity_category shares the 8-band vocabulary with by_capacity.entity"},
    "demand_tariff_mv": {"join_on": "date", "related_to": ["price_with_usd"], "join_note": "join price_with_usd on date for xrate; value is GEL/kWh, so USD/kWh = value / xrate"},
}

# --- Output scrubber ---
def scrub_schema_mentions(text: str) -> str:
    if not text:
        return text
    # Apply DERIVED_LABELS first (longer, more specific names like share_renewable_ppa)
    for col, label in DERIVED_LABELS.items():
        text = re.sub(rf"\b{re.escape(col)}\b", label, text, flags=re.IGNORECASE)
    for col, label in COLUMN_LABELS.items():
        if col in SCRUB_EXEMPT_COLUMNS:
            continue
        text = re.sub(rf"\b{re.escape(col)}\b", label, text, flags=re.IGNORECASE)
    for tbl, label in VIEW_LABELS.items():
        text = re.sub(rf"\b{re.escape(tbl)}\b", label, text, flags=re.IGNORECASE)
    for val, label in VALUE_LABELS.items():
        text = re.sub(rf"\b{re.escape(val)}\b", label, text, flags=re.IGNORECASE)
    schema_terms = ["schema", "table", "column", "sql", "join", "primary key", "foreign key", "view", "constraint"]
    for term in schema_terms:
        text = re.sub(rf"\b{re.escape(term)}\b", "data", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    return text


# Source anchors the summarizer is asked to put in the structured ``citations``
# field — NOT inline. Some models (e.g. gpt-oss-120b) leak them into the answer
# body as ``*[domain_knowledge]*``, ``[statistics]`` or ``【regulated_plant_sales】``.
# The anchor set is OPEN-ENDED: why-context evidence blocks add their own anchors
# (regulated_plant_sales, component_pressure, …), so match the marker SHAPE —
# a known base anchor OR any snake_case identifier with an underscore — rather
# than a fixed vocabulary that silently misses new anchors.
_CITATION_ANCHORS = (
    "data_preview", "statistics", "domain_knowledge",
    "external_source_passages", "conversation_history",
)
# One anchor token: a base anchor, or a snake_case id with ≥1 underscore (covers
# evidence-block anchors and future ones). Excludes prose like [2024] / [note].
_CITATION_ANCHOR_TOKEN = (
    r"(?:" + "|".join(_CITATION_ANCHORS) + r"|[a-z][a-z0-9]*(?:_[a-z0-9]+)+)"
)
_CITATION_MARKER_RE = re.compile(
    r"\s*\*{0,2}[\[\(【]\s*"
    + _CITATION_ANCHOR_TOKEN
    + r"(?:\s*[,;/&]\s*" + _CITATION_ANCHOR_TOKEN + r")*"
    + r"\s*[\]\)】]\*{0,2}",
    re.IGNORECASE,
)


def strip_inline_citation_markers(text: str) -> str:
    """Remove inline citation tags the model leaks into the answer prose.

    Citations belong in the structured ``citations`` field, not the body. Strips
    ``*[domain_knowledge]*``, ``[statistics]``, ``【statistics】`` and multi-anchor
    variants, then tidies the whitespace/punctuation left behind. Anchors are
    technical tokens that don't occur in natural prose, so this is safe.
    """
    if not text:
        return text
    cleaned = _CITATION_MARKER_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)        # collapse double spaces
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)  # space before punctuation
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)        # trailing space before newline
    return cleaned.strip()


# --- Supply/Demand/Transit explicit lists for backend filtering ---
SUPPLY_TECH_TYPES = list(dict.fromkeys(list(TECH_TYPE_GROUPS["supply"].keys()) + ["self-cons"]))
DEMAND_TECH_TYPES = list(TECH_TYPE_GROUPS["demand"].keys())
TRANSIT_TECH_TYPES = ["transit"]
# Domestic generation technologies: the supply side minus non-generation
# sources (import is purchased abroad, self-cons never reaches the grid).
# "Generation mix" questions are about THIS set — supply mix / energy-balance /
# import-dependence questions keep the broader SUPPLY_TECH_TYPES.
GENERATION_TECH_TYPES = [t for t in SUPPLY_TECH_TYPES if t not in ("import", "self-cons")]
