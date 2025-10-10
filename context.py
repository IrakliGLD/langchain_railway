# === context.py v2.0 ===
# Updated for main.py v18.3: schema reinforcement, column alias hints, validation support.

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
    "hydro": "Hydro Generation",
    "thermal": "Thermal Generation",
    "wind": "Wind Generation",
    "solar": "Solar Generation",
    "import": "Imports",
    "export": "Exports",
    "losses": "Grid Losses",
    "abkhazeti": "Abkhazia Consumption",
    "transit": "Transit Flows",
    "HPP": "Hydropower Plant",
    "TPP": "Thermal Power Plant",
    "Solar": "Solar Plant",
    "Wind": "Wind Plant",
    "Import": "Import",
    "overall CPI": "Overall Consumer Price Index",
    "electricity_gas_and_other_fuels": "Electricity, Gas & Other Fuels CPI",
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
    "balancing_electricity": "Balancing Electricity",
    "bilateral_exchange": "Bilateral Contracts & Exchange",
    "renewable_ppa": "Renewable PPA",
    "thermal_ppa": "Thermal PPA",
}

# --- Structured Schema Dict ---
DB_SCHEMA_DICT = {
    "tables": {
        "dates": {"columns": ["date"], "desc": "Calendar (Months)"},
        "energy_balance_long": {
            "columns": ["year", "sector", "energy_source", "volume_tj"],
            "desc": "Energy Balance (by Sector, yearly data)",
        },
        "entities": {"columns": ["entity", "entity_normalized", "type", "ownership", "source"], "desc": "Power Sector Entities"},
        "monthly_cpi": {"columns": ["date", "cpi_type", "cpi"], "desc": "Consumer Price Index (CPI)"},
        "price": {"columns": ["date", "p_dereg_gel", "p_bal_gel", "p_gcap_gel", "xrate"], "desc": "Electricity Market Prices"},
        "tariff_gen": {"columns": ["date", "entity", "tariff_gel"], "desc": "Regulated Tariffs"},
        "tech_quantity": {"columns": ["date", "type_tech", "quantity_tech"], "desc": "Generation & Demand Quantities"},
        "trade": {"columns": ["date", "entity", "segment", "quantity"], "desc": "Electricity Trade"},
    },
    "rules": {
        "unit_conversion": "1 TJ = 277.778 MWh; tech_quantity/trade in thousand MWh (multiply by 1000 for MWh).",
        "usd_rule": "USD values = corresponding GEL value / xrate (from price table, joined by date).",
        "granularity": "Monthly data; energy_balance_long is yearly.",
        "temporal_scope": "2015–present; use full time range.",
    },
}

# --- Reinforced Schema Text for LLM Context (Option 3) ---
DB_SCHEMA_DOC = """
### Key Database Rules and Conventions

**Allowed Columns Only:**
Use only these exact column names when forming SQL:
- date, year, sector, energy_source, volume_tj
- entity, entity_normalized, type, ownership, source
- cpi_type, cpi
- p_dereg_gel, p_bal_gel, p_gcap_gel, xrate
- tariff_gel
- type_tech, quantity_tech
- segment, quantity

Do NOT invent variants like 'quantity_mwh', 'p_bal', or 'tariff_usd' — use correct names and apply /xrate for USD conversion.

**Unit Conversions:**
- 1 TJ = 277.778 MWh
- Quantities in `tech_quantity` and `trade` are thousand MWh → multiply by 1000 for MWh.

**USD Conversions:**
- No *_usd columns exist physically.
- Compute manually:
  - p_dereg_usd = p_dereg_gel / xrate
  - p_bal_usd = p_bal_gel / xrate
  - tariff_usd = tariff_gel / xrate

**Granularity:**
- Monthly for all tables with `date`.
- Yearly for `energy_balance_long` (use column `year`).

**Joins:**
- Always join on `date` or `entity` only when tables share those fields.
- No cross joins or system tables.

**Time Coverage:**
2015–latest month.

**Important Aliases and Clarifications:**
If user asks for:
- “energy in MWh” → use `quantity_tech`
- “tariff in USD” → `tariff_gel / xrate`
- “balancing vs deregulated prices” → `p_bal_gel` and `p_dereg_gel`
- “generation type” → `type_tech`
- “trade volume” → `quantity`

Keep queries safe, aggregated, and small (add LIMIT 500 if needed).
"""

# --- Join Map ---
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

# --- Output scrubber ---
def scrub_schema_mentions(text: str) -> str:
    if not text:
        return text
    for col, label in COLUMN_LABELS.items():
        text = re.sub(rf"\\b{re.escape(col)}\\b", label, text, flags=re.IGNORECASE)
    for tbl, label in TABLE_LABELS.items():
        text = re.sub(rf"\\b{re.escape(tbl)}\\b", label, text, flags=re.IGNORECASE)
    for val, label in VALUE_LABELS.items():
        text = re.sub(rf"\\b{re.escape(val)}\\b", label, text, flags=re.IGNORECASE)
    schema_terms = ["schema", "table", "column", "sql", "join", "primary key", "foreign key", "view", "constraint"]
    for term in schema_terms:
        text = re.sub(rf"\\b{re.escape(term)}\\b", "data", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    if any(re.search(rf"\\b{re.escape(term)}\\b", text, re.IGNORECASE) for term in schema_terms):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", "Remove any technical jargon (schema, table, sql, join) from text while keeping meaning."),
            ("human", text),
        ])
        chain = fallback_prompt | llm | StrOutputParser()
        text = chain.invoke({})
    return text
