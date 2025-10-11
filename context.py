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

    # energy_balance_long_mv
    "year": "Year",
    "sector": "Sector",
    "energy_source": "Energy Source",
    "volume_tj": "Energy Consumption (TJ)",

    # entities_mv
    "entity": "Entity Name",
    "entity_normalized": "Standardized Entity ID",
    "type": "Entity Type",
    "ownership": "Ownership",
    "source": "Source (Local vs Import-Dependent)",

    # monthly_cpi_mv
    "cpi_type": "CPI Category",
    "cpi": "CPI Value (2015=100)",

    # price_with_usd
    "p_dereg_gel": "Deregulated Price (GEL/MWh)",
    "p_bal_gel": "Balancing Price (GEL/MWh)",
    "p_gcap_gel": "Guaranteed Capacity Fee (GEL/MWh)",
    "xrate": "Exchange Rate (GEL/USD)",
    "p_dereg_usd": "Deregulated Price (USD/MWh)",
    "p_bal_usd": "Balancing Price (USD/MWh)",
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

# --- Table label mapping ---
VIEW_LABELS = {
    "energy_balance_long_mv": "Energy Balance (by Sector)",
    "entities_mv": "Power Sector Entities",
    "monthly_cpi_mv": "Consumer Price Index (CPI)",
    "price_with_usd": "Electricity Market Prices (USD)",
    "tariff_with_usd": "Regulated Tariffs (USD)",
    "tech_quantity_view": "Generation & Demand Quantities",
    "trade_derived_entities": "Electricity Trade",
}

# --- Demand/Supply classification for type_tech ---
TECH_TYPE_GROUPS = {
    "demand": {
        "abkhazeti": "Abkhazeti",
        "supply_distribution": "Supplier/Distributor",
        "direct_customers": "Direct Consumers",
        "self_cons": "Self-Consumption by PP",
        "losses": "Losses",
        "export": "Export",
    },
    "supply": {
        "hydro": "Hydro Generation",
        "thermal": "Thermal Generation",
        "wind": "Wind Generation",
        "import": "Import",
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
    "overall CPI": "Overall Consumer Price Index",
    "electricity_gas_and_other_fuels": "Electricity, Gas & Other Fuels CPI",
    "Coal": "Coal",
    "Oil products": "Oil Products",
    "Natural Gas": "Natural Gas",
    "Biofuel & Waste": "Biofuel & Waste",
    "Electricity": "Electricity",
    "Heat": "Heat",
    "Total": "Total Energy Use",
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
        "energy_balance_long_mv": {
            "columns": ["year", "sector", "energy_source", "volume_tj"],
            "desc": "Energy Balance (by Sector, yearly data)",
        },
        "entities_mv": {
            "columns": ["entity", "entity_normalized", "type", "ownership", "source"],
            "desc": "Power Sector Entities",
        },
        "monthly_cpi_mv": {
            "columns": ["date", "cpi_type", "cpi"],
            "desc": "Consumer Price Index (CPI)",
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
        "unit_conversion": "1 TJ = 277.778 MWh; tech_quantity/trade data in thousand MWh (×1000 for MWh).",
        "usd_rule": "USD values = GEL / xrate (from price_with_usd joined by date).",
        "granularity": "Monthly data for all except yearly energy_balance_long_mv.",
        "temporal_scope": "2015–present; use full range.",
    },
}

# --- Reinforced Schema Text for LLM Context ---
DB_SCHEMA_DOC = """
### Key Database Rules and Conventions (Materialized Views)

**Available Views:**
- energy_balance_long_mv(year, sector, energy_source, volume_tj)
- entities_mv(entity, entity_normalized, type, ownership, source)
- monthly_cpi_mv(date, cpi_type, cpi)
- price_with_usd(date, p_dereg_gel, p_bal_gel, p_gcap_gel, xrate, p_dereg_usd, p_bal_usd, p_gcap_usd)
- tariff_with_usd(date, entity, tariff_gel, tariff_usd)
- tech_quantity_view(date, type_tech, quantity_tech)
- trade_derived_entities(date, entity, segment, quantity)

**Type_tech classification:**
- Demand: abkhazeti, supply_distribution, direct_customers, self_cons, losses, export
- Supply: hydro, thermal, wind, import

**Units & Conversions:**
- 1 TJ = 277.778 MWh
- Quantities in thousand MWh (multiply ×1000 for MWh)
- *_usd fields = *_gel / xrate

**Granularity:**
- Monthly for all except energy_balance_long_mv (yearly)

**Joins:**
- Use date or entity as join keys only.
- Avoid system tables or undefined joins.

**Time Coverage:** 2015–latest month
"""

# --- Join Map ---
DB_JOINS = {
    "energy_balance_long_mv": {"join_on": "year", "related_to": ["tech_quantity_view", "price_with_usd", "monthly_cpi_mv"]},
    "price_with_usd": {"join_on": "date", "related_to": ["tariff_with_usd", "tech_quantity_view", "trade_derived_entities", "monthly_cpi_mv"]},
    "tariff_with_usd": {"join_on": ["date", "entity"], "related_to": ["price_with_usd", "trade_derived_entities"]},
    "tech_quantity_view": {"join_on": "date", "related_to": ["price_with_usd", "trade_derived_entities", "monthly_cpi_mv"]},
    "trade_derived_entities": {"join_on": ["date", "entity"], "related_to": ["price_with_usd", "tariff_with_usd"]},
    "monthly_cpi_mv": {"join_on": "date", "related_to": ["price_with_usd", "tech_quantity_view", "energy_balance_long_mv"]},
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
