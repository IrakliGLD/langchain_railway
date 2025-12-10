"""
Application configuration and constants.

Extracts all environment variables, constants, and configuration
from the monolithic main.py for better organization.
"""
import os
import re
from textwrap import dedent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===================================================================
# Environment Variables
# ===================================================================

# LLM API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# API Security
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

# LLM Configuration
MODEL_TYPE = os.getenv("MODEL_TYPE", "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Query Limits
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))

# ===================================================================
# Validation
# ===================================================================

if not SUPABASE_DB_URL:
    raise RuntimeError("Missing SUPABASE_DB_URL")
if not APP_SECRET_KEY:
    raise RuntimeError("Missing APP_SECRET_KEY")
if MODEL_TYPE == "gemini" and not GOOGLE_API_KEY:
    raise RuntimeError("MODEL_TYPE=gemini but GOOGLE_API_KEY is missing")

# ===================================================================
# Database Configuration
# ===================================================================

# Allowed tables for SQL validation (whitelist)
STATIC_ALLOWED_TABLES = {
    "dates_mv",
    "entities_mv",
    "price_with_usd",
    "tariff_with_usd",
    "tech_quantity_view",
    "trade_derived_entities",
    "monthly_cpi_mv",
    "energy_balance_long_mv"
}

ALLOWED_TABLES = set(STATIC_ALLOWED_TABLES)

# Table synonyms for auto-correction
TABLE_SYNONYMS = {
    "prices": "price",
    "tariffs": "tariff_gen",
    "price_usd": "price_with_usd",
    "tariff_usd": "tariff_with_usd",
    "price_with_usd": "price_with_usd",
}

# Column synonyms for auto-correction
COLUMN_SYNONYMS = {
    "tech_type": "type_tech",
    "quantity_mwh": "quantity_tech",
}

# Pre-compiled regex patterns for SQL synonym replacement (performance)
SYNONYM_PATTERNS = [
    (re.compile(r"\bprices\b", re.IGNORECASE), "price_with_usd"),
    (re.compile(r"\btariffs\b", re.IGNORECASE), "tariff_with_usd"),
    (re.compile(r"\btech_quantity\b", re.IGNORECASE), "tech_quantity_view"),
    (re.compile(r"\btrade\b", re.IGNORECASE), "trade_derived_entities"),
    (re.compile(r"\bentities\b", re.IGNORECASE), "entities_mv"),
    (re.compile(r"\bmonthly_cpi\b", re.IGNORECASE), "monthly_cpi_mv"),
    (re.compile(r"\benergy_balance_long\b", re.IGNORECASE), "energy_balance_long_mv"),
]

# Pre-compiled regex for LIMIT detection
LIMIT_PATTERN = re.compile(r"\bLIMIT\s*\d+\b", re.IGNORECASE)

# ===================================================================
# Analysis Configuration
# ===================================================================

# Seasonal months
SUMMER_MONTHS = [4, 5, 6, 7]
WINTER_MONTHS = [1, 2, 3, 8, 9, 10, 11, 12]

# Balancing segment normalizer
BALANCING_SEGMENT_NORMALIZER = "LOWER(REPLACE(segment, ' ', '_'))"

# Balancing share pivot SQL template
BALANCING_SHARE_PIVOT_SQL = dedent(
    f"""
    SELECT
        t.date,
        'balancing'::text AS segment,
        SUM(t.quantity) AS total_quantity_debug,
        SUM(CASE WHEN t.entity = 'import' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_import,
        SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_deregulated_hydro,
        SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_regulated_hpp,
        SUM(CASE WHEN t.entity = 'regulated_new_tpp' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_regulated_new_tpp,
        SUM(CASE WHEN t.entity = 'regulated_old_tpp' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_regulated_old_tpp,
        SUM(CASE WHEN t.entity = 'renewable_ppa' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_renewable_ppa,
        SUM(CASE WHEN t.entity = 'thermal_ppa' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_thermal_ppa,
        SUM(CASE WHEN t.entity IN ('renewable_ppa','thermal_ppa') THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_all_ppa,
        SUM(CASE WHEN t.entity IN ('deregulated_hydro','regulated_hpp','renewable_ppa') THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_all_renewables,
        SUM(CASE WHEN t.entity IN ('deregulated_hydro','regulated_hpp') THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_total_hpp
    FROM trade_derived_entities t
    WHERE {BALANCING_SEGMENT_NORMALIZER} = 'balancing'
      AND t.entity IN (
        'import', 'deregulated_hydro', 'regulated_hpp',
        'regulated_new_tpp', 'regulated_old_tpp',
        'renewable_ppa', 'thermal_ppa'
      )
    GROUP BY t.date
    ORDER BY t.date
    """
).strip()

# Balancing share metadata
BALANCING_SHARE_METADATA = {
    "share_import": {"label": "Import"},
    "share_deregulated_hydro": {"label": "Deregulated Hydro"},
    "share_regulated_hpp": {"label": "Regulated HPP"},
    "share_regulated_new_tpp": {"label": "Regulated New TPP"},
    "share_regulated_old_tpp": {"label": "Regulated Old TPP"},
    "share_renewable_ppa": {"label": "Renewable PPA"},
    "share_thermal_ppa": {"label": "Thermal PPA"},
    "share_all_ppa": {"label": "All PPAs"},
    "share_all_renewables": {"label": "All Renewables"},
    "share_total_hpp": {"label": "Total HPP"},
}

# ===================================================================
# Cache Configuration
# ===================================================================

CACHE_MAX_SIZE = 1000
CACHE_EVICTION_PERCENT = 0.1

# ===================================================================
# SQL Security
# ===================================================================

SQL_TIMEOUT_SECONDS = 30
DATABASE_POOL_SIZE = 10
DATABASE_MAX_OVERFLOW = 5
