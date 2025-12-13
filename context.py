# === context.py v2.1 ===
# Updated for main.py v18.5: materialized-view schema, OpenAI context alignment, demand/supply groups.

import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Column label mapping ---
COLUMN_LABELS = {
    # shared
    "date": "Period (Year-Month-Day)",


    # entities_mv
    "entity": "Entity Name",
    "entity_normalized": "Standardized Entity ID",
    "type": "Entity Type",
    "ownership": "Ownership",
    "source": "Source (Local vs Import-Dependent)",


    # price_with_usd
    "p_dereg_gel": "Deregulated Price (GEL/MWh)",
    "p_bal_gel": "Balancing Electricity Price (GEL/MWh)",
    "p_gcap_gel": "Guaranteed Capacity Fee (GEL/MWh)",
    "xrate": "Exchange Rate (GEL/USD)",
    "p_dereg_usd": "Deregulated Price (USD/MWh)",
    "p_bal_usd": "Balancing Electricity Price (USD/MWh)",
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
}

# ----------------------------------------------------------
# DERIVED_LABELS — for LLM-generated / computed columns
# (These do not exist physically in Supabase views)
# ----------------------------------------------------------
DERIVED_LABELS = {
    "share_import": "Share of Imports in Balancing Electricity",
    "share_cfd_scheme": "Share of CfD scheme supported generation in Balancing Electricity",
    "share_deregulated_hydro": "Share of Deregulated Hydro",
    "share_regulated_hpp": "Share of Regulated HPPs",
    "share_regulated_new_tpp": "Share of Regulated New TPPs",
    "share_regulated_old_tpp": "Share of Regulated Old TPPs",
    "share_total_hpp": "Share of Total HPP Output",
    "share_renewable_ppa": "Share of Renewable PPAs",
    "share_thermal_ppa": "Share of Thermal PPAs",
    "share_all_ppa": "Share of All PPAs",
    "share_all_renewables": "Share of All Renewable Sources",
    "enguri_tariff_gel": "Enguri HPP Tariff (GEL/MWh)",
    "gardabani_tpp_tariff_gel": "Gardabani TPP Tariff (GEL/MWh)",
    "grouped_old_tpp_tariff_gel": "Old Thermal Power Plants Tariff (GEL/MWh)",
    "weighted_gel": "Weighted-Average Balancing Price (GEL/MWh)",
    "weighted_usd": "Weighted-Average Balancing Price (USD/MWh)",
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

# --- Value label mapping ---
VALUE_LABELS = {
    **TECH_TYPE_GROUPS["demand"],
    **TECH_TYPE_GROUPS["supply"],
    "solar": "Solar Generation",
    "transit": "Transit Flows",
    "HPP": "Hydropower Plant",
    "TPP": "Thermal Power Plant",
    "balancing_electricity": "Balancing Electricity",
    "bilateral_exchange": "Bilateral Contracts & Exchange",
    "renewable_ppa": "Renewable PPA",
    "thermal_ppa": "Thermal PPA",
    "deregulated_hydro": "Deregulated HPP",
    "regulated_hpp": "Regulated HPP",
    "regulated_new_tpp": "Regulated new TPP",
    "regulated_old_tpp": "Regulated old TPP",
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
- tech_quantity_view(date, type_tech, quantity_tech)
- trade_derived_entities(date, entity, segment, quantity)
- trade_by_source (date,source,quantity_tech)

**CRITICAL: Exact column values (case-sensitive, including spaces/hyphens):**

type_tech values (tech_quantity_view):
- Demand side: 'abkhazeti', 'supply-distribution' (note: hyphen!), 'direct customers' (note: space! - MARKET CATEGORY not industry sector, see DirectCustomers domain knowledge), 'losses', 'export'
- Supply side: 'hydro', 'thermal', 'wind', 'import', 'solar'
- IMPORTANT: Use exact strings with hyphens and spaces as shown above!

segment values (trade_derived_entities):
- 'Balancing Electricity' (note: capital B, capital E, with space!)
- 'Bilateral Contracts & Exchange'
- IMPORTANT: Segment values are case-sensitive and may contain spaces!
- Common mistake: using lowercase 'balancing' - WRONG! Use 'Balancing Electricity'
- Recommended filter: WHERE LOWER(REPLACE(segment, ' ', '_')) = 'balancing' (handles case variations)

**Units & Conversions:**
- Quantities in thousand MWh (multiply ×1000 for MWh)
- *_usd fields = *_gel / xrate

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
"""

# --- Join Map ---
DB_JOINS = {
    "price_with_usd": {"join_on": "date", "related_to": ["tariff_with_usd", "tech_quantity_view", "trade_derived_entities"]},
    "tariff_with_usd": {"join_on": ["date", "entity"], "related_to": ["price_with_usd", "trade_derived_entities"]},
    "tech_quantity_view": {"join_on": "date", "related_to": ["price_with_usd", "trade_derived_entities"]},
    "trade_derived_entities": {"join_on": ["date", "entity"], "related_to": ["price_with_usd", "tariff_with_usd"]},
    "entities_mv": {"join_on": "entity", "related_to": ["tariff_with_usd", "trade_derived_entities"]},
}

# --- Output scrubber ---
def scrub_schema_mentions(text: str) -> str:
    if not text:
        return text
    for col, label in COLUMN_LABELS.items():
        text = re.sub(rf"\b{re.escape(col)}\b", label, text, flags=re.IGNORECASE)
    for tbl, label in VIEW_LABELS.items():
        text = re.sub(rf"\b{re.escape(tbl)}\b", label, text, flags=re.IGNORECASE)
    for val, label in VALUE_LABELS.items():
        text = re.sub(rf"\b{re.escape(val)}\b", label, text, flags=re.IGNORECASE)
    schema_terms = ["schema", "table", "column", "sql", "join", "primary key", "foreign key", "view", "constraint"]
    for term in schema_terms:
        text = re.sub(rf"\b{re.escape(term)}\b", "data", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    if any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in schema_terms):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", "Remove technical jargon (schema, SQL, join, etc.) while keeping meaning clear."),
            ("human", text),
        ])
        chain = fallback_prompt | llm | StrOutputParser()
        text = chain.invoke({})
    return text

# --- Supply/Demand/Transit explicit lists for backend filtering ---
SUPPLY_TECH_TYPES = list(TECH_TYPE_GROUPS["supply"].keys()) + ["solar", "self-cons"]
DEMAND_TECH_TYPES = list(TECH_TYPE_GROUPS["demand"].keys())
TRANSIT_TECH_TYPES = ["transit"]
