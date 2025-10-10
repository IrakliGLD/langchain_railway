# === context.py v1.9 ===
# Added: Monthly granularity, temporal coverage clarifications, and note that energy_balance_long is yearly.
# Preserved USD/xrate conversion logic from v1.8.

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
    "date": "Period (Year-Month)",

    # energy_balance_long
    "year": "Year",
    "sector": "Sector",
    "energy_source": "Energy Source",
    "volume_tj": "Energy Consumption (TJ)",

    # entities
    "entity": "Entity Name",
    "entity_normalized": "Standardized Entity ID",
    "type": "Entity Type",
    "ownership": "Ownership",
    "source": "Source (Local vs Import-Dependent)",

    # monthly_cpi
    "cpi_type": "CPI Category",
    "cpi": "CPI Value (2015=100)",

    # price
    "p_dereg_gel": "Deregulated Price (GEL/MWh)",
    "p_bal_gel": "Balancing Price (GEL/MWh)",
    "p_gcap_gel": "Guaranteed Capacity Fee (GEL/MWh)",
    "xrate": "Exchange Rate (GEL/USD)",
    "p_dereg_usd": "Deregulated Price (USD/MWh)",
    "p_bal_usd": "Balancing Price (USD/MWh)",
    "p_gcap_usd": "Guaranteed Capacity Fee (USD/MWh)",

    # tariff_gen
    "tariff_gel": "Regulated Tariff (GEL/MWh)",
    "tariff_usd": "Regulated Tariff (USD/MWh)",

    # tech_quantity
    "type_tech": "Technology Type",
    "quantity_tech": "Quantity (thousand MWh)",

    # trade
    "segment": "Market Segment",
    "quantity": "Trade Volume (thousand MWh)",
}

# --- Table label mapping ---
TABLE_LABELS = {
    "dates": "Calendar (Months)",
    "energy_balance_long": "Energy Balance (by Sector)",
    "entities": "Power Sector Entities",
    "monthly_cpi": "Consumer Price Index (CPI)",
    "price": "Electricity Market Prices",
    "tariff_gen": "Regulated Tariffs",
    "tech_quantity": "Generation & Demand Quantities",
    "trade": "Electricity Trade",
}

# --- Value label mapping ---
VALUE_LABELS = {
    # tech_quantity.type_tech
    "hydro": "Hydro Generation",
    "thermal": "Thermal Generation",
    "wind": "Wind Generation",
    "solar": "Solar Generation",
    "import": "Imports",
    "export": "Exports",
    "losses": "Grid Losses",
    "abkhazeti": "Abkhazia Consumption",
    "transit": "Transit Flows",

    # entities.type
    "HPP": "Hydropower Plant",
    "TPP": "Thermal Power Plant",
    "Solar": "Solar Plant",
    "Wind": "Wind Plant",
    "Import": "Import",

    # CPI categories
    "overall CPI": "Overall Consumer Price Index",
    "electricity_gas_and_other_fuels": "Electricity, Gas & Other Fuels CPI",

    # energy sources in energy_balance_long
    "Coal": "Coal",
    "Oil products": "Oil Products",
    "Natural Gas": "Natural Gas",
    "Hydro": "Hydropower",
    "Wind": "Wind Power",
    "Solar": "Solar Power",
    "Biofuel & Waste": "Biofuel & Waste",
    "Electricity": "Electricity",
    "Heat": "Heat",
    "Total": "Total Energy Use",

    # trade.segment values
    "balancing_electricity": "Balancing Electricity",
    "bilateral_exchange": "Bilateral Contracts & Exchange",
    "renewable_ppa": "Renewable PPA",
    "thermal_ppa": "Thermal PPA",
}

# --- Structured Schema Dict (for validation) ---
DB_SCHEMA_DICT = {
    "tables": {
        "dates": {"columns": ["date"], "desc": "Calendar (Months)"},
        # energy_balance_long contains yearly data (dimension = year)
        "energy_balance_long": {"columns": ["year", "sector", "energy_source", "volume_tj"], "desc": "Energy Balance (by Sector, yearly data)"},
        "entities": {"columns": ["entity", "entity_normalized", "type", "ownership", "source"], "desc": "Power Sector Entities"},
        "monthly_cpi": {"columns": ["date", "cpi_type", "cpi"], "desc": "Consumer Price Index (CPI)"},
        "price": {"columns": ["date", "p_dereg_gel", "p_bal_gel", "p_gcap_gel", "xrate"], "desc": "Electricity Market Prices"},
        "tariff_gen": {"columns": ["date", "entity", "tariff_gel"], "desc": "Regulated Tariffs"},
        "tech_quantity": {"columns": ["date", "type_tech", "quantity_tech"], "desc": "Generation & Demand Quantities"},
        "trade": {"columns": ["date", "entity", "segment", "quantity"], "desc": "Electricity Trade"},
    },
    "rules": {
        "unit_conversion": "1 TJ = 277.778 MWh; tech_quantity/trade in thousand MWh (multiply by 1000 for MWh).",
        "usd_rule": "USD and USD/MWh variables are derived as (corresponding GEL value / xrate). Example: p_bal_usd = p_bal_gel / xrate, tariff_usd = tariff_gel / xrate. Always use same month’s xrate from price table, joined by date.",
        "granularity": "Monthly data (first day of month, YYYY-MM-01 format). Do not treat as daily; each record represents one month. energy_balance_long contains yearly data (dimension = year).",
        "temporal_scope": "Data available from 2015 up to latest month recorded; analyses must use full time range, not only earliest entries.",
        "forecast_restriction": "Only for prices, CPI, demand; no generation/imports/exports.",
    },
}

# --- Prose Schema Doc (for LLM context) ---
DB_SCHEMA_DOC = """
### Global Rules & Conversions ###
- **General Rule:** Provide summaries and insights only. Do NOT return raw data, full tables, or row-level dumps. If asked for a dump, refuse and suggest an aggregated view instead.

- **Unit Conversion:** 
  - 1 TJ = 277.778 MWh
  - The `tech_quantity` and `trade` tables store quantities in **thousand MWh**; multiply by 1000 for MWh.

- **USD Conversion Rule:**
  - Any variable ending with `_usd` or described in USD/MWh does **not exist directly** in the database.
  - These must be derived as:
    - `p_dereg_usd = p_dereg_gel / xrate`
    - `p_bal_usd = p_bal_gel / xrate`
    - `p_gcap_usd = p_gcap_gel / xrate`
    - `tariff_usd = tariff_gel / xrate`
  - Always use the **same month’s `xrate`** for conversion — join on the `date` column between the target table and the `price` table.
  - ⚠️ *Note:* There is **no separate `xrates` table.**
    The exchange rate column (`xrate`) is located **inside the `price` table**.
    When performing conversions, always use `price.xrate` joined by the same `date`.

- **Monthly Granularity:**
  - All tables with a `date` column contain **monthly** data stored as the **first day of the month (YYYY-MM-01)**.
  - Treat each record as one full month, not a daily entry.
  - Example: a record with date '2024-05-01' represents data for **May 2024**.

- **Yearly Data:**
  - The `energy_balance_long` table uses a **year** dimension (not monthly).
  - Its data is aggregated annually by sector and energy source.

- **Temporal Coverage:**
  - Data spans from **2015 to the most recent available month**.
  - Analyses (e.g., trends, comparisons) must use the **full available range**, not just early data like 2015.

- **Forecasting Restriction:**
  - Forecasts can be made for prices, CPI, and demand.
  - For generation (hydro, thermal, wind, solar) and imports/exports: only historical trends can be shown.
  - Future projections depend on new capacity/projects not included in this dataset.
"""

# --- DB_JOINS v1.4 ---
DB_JOINS = {
    "dates": {"join_on": "date", "related_to": ["price", "monthly_cpi", "tech_quantity", "trade", "tariff_gen"]},
    "energy_balance_long": {"join_on": "year", "related_to": ["tech_quantity", "price", "monthly_cpi", "trade"]},
    "price": {"join_on": "date", "related_to": ["trade", "tech_quantity", "monthly_cpi", "energy_balance_long"]},
    "trade": {"join_on": ["date", "entity"], "related_to": ["price", "entities", "tariff_gen", "tech_quantity", "energy_balance_long"]},
    "tech_quantity": {"join_on": "date", "related_to": ["price", "energy_balance_long", "trade", "monthly_cpi"]},
    "tariff_gen": {"join_on": ["date", "entity"], "related_to": ["entities", "trade", "price", "tech_quantity"]},
    "entities": {"join_on": "entity", "related_to": ["tariff_gen", "trade"]},
    "monthly_cpi": {"join_on": "date", "related_to": ["price", "tech_quantity", "energy_balance_long"]},
}

# --- Human-friendly scrubber (labels-aware) ---
def scrub_schema_mentions(text: str) -> str:
    """
    Cleans final model output so that:
    - Raw SQL/schema terms are humanized.
    - Column names -> user-friendly labels.
    - Table names -> user-friendly labels.
    - Encoded categorical values -> natural labels.
    """
    if not text:
        return text

    # 1) Columns -> labels
    for col, label in COLUMN_LABELS.items():
        text = re.sub(rf"\\b{re.escape(col)}\\b", label, text, flags=re.IGNORECASE)

    # 2) Tables -> labels
    for tbl, label in TABLE_LABELS.items():
        text = re.sub(rf"\\b{re.escape(tbl)}\\b", label, text, flags=re.IGNORECASE)

    # 3) Encoded values -> natural labels
    for val, label in VALUE_LABELS.items():
        text = re.sub(rf"\\b{re.escape(val)}\\b", label, text, flags=re.IGNORECASE)

    # 4) Hide schema/SQL jargon
    schema_terms = ["schema", "table", "column", "sql", "join", "primary key", "foreign key", "view", "constraint"]
    for term in schema_terms:
        text = re.sub(rf"\\b{re.escape(term)}\\b", "data", text, flags=re.IGNORECASE)

    # 5) Strip markdown fences
    text = text.replace("```", "").strip()

    # 6) LLM fallback if terms still present
    if any(re.search(rf"\\b{re.escape(term)}\\b", text, re.IGNORECASE) for term in schema_terms):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", "Remove any technical jargon like schema, table, sql, join from text. Keep meaning intact."),
            ("human", text),
        ])
        chain = fallback_prompt | llm | StrOutputParser()
        text = chain.invoke({})

    return text
