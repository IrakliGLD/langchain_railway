"""
Pipeline Stage 2: SQL Validation & Execution

Validates LLM-generated SQL, applies safety checks, pivot injection,
executes against the database, and handles error recovery.
"""
import logging
import re
import time
from typing import Optional, Tuple

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError, OperationalError, SQLAlchemyError

from models import QueryContext
from core.sql_generator import sanitize_sql, simple_table_whitelist_check, plan_validate_repair
from core.query_executor import ENGINE, execute_sql_safely
from agent.aggregation import validate_aggregation_logic
from agent.provenance import clear_provenance, sql_query_hash, stamp_provenance
from utils.trace_logging import trace_detail
from utils.query_validation import validate_sql_relevance

log = logging.getLogger("Enai")


# ---------------------------------------------------------------------------
# SQL column synonym auto-fix (from context.py)
# ---------------------------------------------------------------------------
try:
    from context import COLUMN_SYNONYMS
except ImportError:
    COLUMN_SYNONYMS = {}

# ---------------------------------------------------------------------------
# Tech type classifications (from context.py)
# ---------------------------------------------------------------------------
try:
    from context import SUPPLY_TECH_TYPES, DEMAND_TECH_TYPES, TRANSIT_TECH_TYPES
except ImportError:
    SUPPLY_TECH_TYPES = ["hydro", "thermal", "wind", "solar", "import", "self-cons"]
    DEMAND_TECH_TYPES = ["abkhazeti", "supply-distribution", "direct customers", "losses", "export"]
    TRANSIT_TECH_TYPES = ["transit"]


# ---------------------------------------------------------------------------
# Constants (moved from main.py)
# ---------------------------------------------------------------------------

BALANCING_SEGMENT_NORMALIZER = "balancing"

BALANCING_SHARE_PIVOT_SQL = f"""
SELECT
    date,
    '{BALANCING_SEGMENT_NORMALIZER}' AS segment,
    MAX(CASE WHEN entity='import' THEN share ELSE 0 END) AS share_import,
    MAX(CASE WHEN entity='deregulated_hydro' THEN share ELSE 0 END) AS share_deregulated_hydro,
    MAX(CASE WHEN entity='regulated_hpp' THEN share ELSE 0 END) AS share_regulated_hpp,
    MAX(CASE WHEN entity='regulated_new_tpp' THEN share ELSE 0 END) AS share_regulated_new_tpp,
    MAX(CASE WHEN entity='regulated_old_tpp' THEN share ELSE 0 END) AS share_regulated_old_tpp,
    MAX(CASE WHEN entity='renewable_ppa' THEN share ELSE 0 END) AS share_renewable_ppa,
    MAX(CASE WHEN entity='thermal_ppa' THEN share ELSE 0 END) AS share_thermal_ppa,
    MAX(CASE WHEN entity='renewable_ppa' THEN share ELSE 0 END) +
        MAX(CASE WHEN entity='thermal_ppa' THEN share ELSE 0 END) AS share_all_ppa,
    MAX(CASE WHEN entity IN ('regulated_hpp','deregulated_hydro','renewable_ppa') THEN share ELSE 0 END) AS share_all_renewables,
    MAX(CASE WHEN entity IN ('regulated_hpp','deregulated_hydro') THEN share ELSE 0 END) AS share_total_hpp
FROM (
    SELECT date, entity,
           ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4) AS share
    FROM trade_derived_entities
    WHERE LOWER(REPLACE(segment, ' ', '_')) = '{BALANCING_SEGMENT_NORMALIZER}'
    GROUP BY date, entity
) sub
GROUP BY date
ORDER BY date DESC
LIMIT 120
""".strip()


# ---------------------------------------------------------------------------
# Helpers (moved from main.py top-level)
# ---------------------------------------------------------------------------

def should_inject_balancing_pivot(user_query: str, sql: str) -> bool:
    """Detect if query is asking for balancing share but SQL doesn't include share calculations."""
    query_lower = user_query.lower()
    sql_lower = sql.lower()

    balancing_keywords = ["balancing", "share", "composition", "mix", "weight", "proportion"]
    entity_keywords = ["ppa", "renewable", "thermal", "import", "hydro", "tpp", "hpp", "entity", "entities"]

    has_balancing = any(k in query_lower for k in balancing_keywords)
    has_entity = any(k in query_lower for k in entity_keywords)
    has_trade = "trade_derived_entities" in sql_lower
    has_share_col = any(f"share_{e}" in sql_lower for e in ["import", "renewable", "ppa", "hydro", "tpp", "hpp"])

    return has_balancing and has_entity and has_trade and not has_share_col


def build_trade_share_cte(original_sql: str) -> str:
    """Inject a balancing electricity share pivot as a CTE and alias original SQL to it."""
    cte_name = "tde"
    cte = f"""WITH {cte_name} AS (
    SELECT
        date,
        entity,
        SUM(quantity) AS quantity,
        ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4) AS share,
        MAX(CASE WHEN entity='import' THEN ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4) ELSE 0 END) OVER (PARTITION BY date) AS share_import,
        MAX(CASE WHEN entity='deregulated_hydro' THEN ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4) ELSE 0 END) OVER (PARTITION BY date) AS share_deregulated_hydro,
        MAX(CASE WHEN entity='regulated_hpp' THEN ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4) ELSE 0 END) OVER (PARTITION BY date) AS share_regulated_hpp,
        MAX(CASE WHEN entity='regulated_new_tpp' THEN ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4) ELSE 0 END) OVER (PARTITION BY date) AS share_regulated_new_tpp,
        MAX(CASE WHEN entity='regulated_old_tpp' THEN ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4) ELSE 0 END) OVER (PARTITION BY date) AS share_regulated_old_tpp,
        MAX(CASE WHEN entity='renewable_ppa' THEN ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4) ELSE 0 END) OVER (PARTITION BY date) AS share_renewable_ppa,
        MAX(CASE WHEN entity='thermal_ppa' THEN ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4) ELSE 0 END) OVER (PARTITION BY date) AS share_thermal_ppa,
        (MAX(CASE WHEN entity='renewable_ppa' THEN ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4) ELSE 0 END) OVER (PARTITION BY date) +
         MAX(CASE WHEN entity='thermal_ppa' THEN ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4) ELSE 0 END) OVER (PARTITION BY date)) AS share_all_ppa,
        (MAX(CASE WHEN entity IN ('regulated_hpp','deregulated_hydro','renewable_ppa') THEN ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4) ELSE 0 END) OVER (PARTITION BY date)) AS share_all_renewables
    FROM trade_derived_entities
    WHERE LOWER(REPLACE(segment, ' ', '_')) = 'balancing'
    GROUP BY date, entity
)
"""
    # Replace trade_derived_entities references in original SQL
    rewritten = re.sub(
        r'\btrade_derived_entities\b',
        cte_name,
        original_sql,
        flags=re.IGNORECASE
    )

    # If original has WITH clause, merge
    if rewritten.strip().upper().startswith("WITH"):
        rewritten = rewritten.strip()[4:]  # Remove leading WITH
        cte = cte.rstrip() + ",\n"

    return cte + rewritten


def fetch_balancing_share_panel(conn) -> pd.DataFrame:
    """Return a DataFrame with monthly balancing share ratios for each entity group."""
    result = conn.execute(text(BALANCING_SHARE_PIVOT_SQL))
    rows = result.fetchall()
    cols = list(result.keys())
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()


def ensure_share_dataframe(
    df: Optional[pd.DataFrame], conn
) -> Tuple[pd.DataFrame, bool]:
    """Ensure we have a dataframe containing share_* columns for summarisation.

    Returns the dataframe to use plus a flag indicating whether the deterministic
    pivot fallback was executed.
    """
    if df is not None and not df.empty:
        share_cols = [c for c in df.columns if c.startswith("share_")]
        if share_cols:
            return df, False

    # Fallback: run deterministic pivot
    fallback_df = fetch_balancing_share_panel(conn)
    return fallback_df, True


def _extract_sql_tables(sql: str) -> list[str]:
    matches = re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w.]*)", sql or "", flags=re.IGNORECASE)
    return sorted(dict.fromkeys(match.lower() for match in matches))


# ---------------------------------------------------------------------------
# Main pipeline stage
# ---------------------------------------------------------------------------

def validate_and_execute(ctx: QueryContext) -> QueryContext:
    """Stage 2: Validate SQL, execute it, and handle errors.

    Reads: ctx.raw_sql, ctx.plan, ctx.query, ctx.skip_sql, ctx.aggregation_intent
    Writes: ctx.safe_sql, ctx.df, ctx.rows, ctx.cols, ctx.sql_is_relevant,
            ctx.skip_chart_due_to_relevance
    """
    if ctx.skip_sql:
        ctx.df = pd.DataFrame()
        ctx.rows = []
        ctx.cols = []
        clear_provenance(ctx)
        trace_detail(
            log,
            ctx,
            "stage_2_sql_execute",
            "skipped",
            reason=ctx.skip_sql_reason or "skip_sql_flag",
        )
        log.info("⏭️ SQL execution skipped, will answer from domain knowledge")
        return ctx

    # --- Validate and sanitize SQL ---
    sanitized = sanitize_sql(ctx.raw_sql.strip())
    simple_table_whitelist_check(sanitized)
    safe_sql = plan_validate_repair(sanitized)

    # Validate aggregation logic
    is_valid_aggregation, validation_reason = validate_aggregation_logic(safe_sql, ctx.aggregation_intent)
    if not is_valid_aggregation:
        log.warning(f"⚠️ SQL doesn't match aggregation intent: {validation_reason}")
    else:
        log.info(f"✅ SQL validation passed: {validation_reason}")

    # Force pivot injection for balancing share queries
    if should_inject_balancing_pivot(ctx.query, safe_sql):
        log.info("🔄 Force-injecting balancing share pivot based on query intent")
        safe_sql = build_trade_share_cte(safe_sql)

    # Period aggregation detection
    period_pattern = re.search(
        r"(?P<start>(?:19|20)\d{2}[-/]?\d{0,2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"[\s–\-to]+"
        r"(?P<end>(?:19|20)\d{2}[-/]?\d{0,2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        ctx.query.lower()
    )

    if period_pattern:
        log.info("🧮 Detected user-defined period range → applying aggregation logic.")
        lower_sql = safe_sql.lower()
        has_agg = any(x in lower_sql for x in ["avg(", "sum(", "count(", "group by"])
        if has_agg:
            log.info("🧮 Query already aggregated → skipping outer AVG/SUM wrapper.")
        else:
            log.warning("⚠️ Period aggregation requested but wrapper is disabled")

    ctx.safe_sql = safe_sql
    trace_detail(
        log,
        ctx,
        "stage_2_sql_execute",
        "sql_ready",
        sql_hash=sql_query_hash(safe_sql),
        tables=_extract_sql_tables(safe_sql),
        aggregation_valid=is_valid_aggregation,
        aggregation_reason=validation_reason,
    )
    trace_detail(
        log,
        ctx,
        "stage_2_sql_execute",
        "artifact",
        debug=True,
        safe_sql=safe_sql,
    )

    # Validate SQL relevance
    ctx.sql_is_relevant, relevance_reason, ctx.skip_chart_due_to_relevance = validate_sql_relevance(
        ctx.query, safe_sql, ctx.plan
    )
    if not ctx.sql_is_relevant:
        log.warning(f"⚠️ SQL relevance issue: {relevance_reason}")
    if ctx.skip_chart_due_to_relevance:
        log.info(f"📊 Chart will be skipped due to: {relevance_reason}")
    if (not ctx.sql_is_relevant) and ctx.skip_chart_due_to_relevance:
        from utils.metrics import metrics

        metrics.log_relevance_block()
        ctx.skip_sql = True
        ctx.skip_sql_reason = f"sql_relevance_blocked:{relevance_reason}"
        ctx.df = pd.DataFrame()
        ctx.rows = []
        ctx.cols = []
        clear_provenance(ctx)
        log.warning("🚫 Blocking SQL execution due to hard relevance policy: %s", relevance_reason)
        trace_detail(
            log,
            ctx,
            "stage_2_sql_execute",
            "blocked",
            reason=relevance_reason,
            skip_chart_due_to_relevance=ctx.skip_chart_due_to_relevance,
        )
        return ctx

    # --- Execute SQL ---
    try:
        log.info(f"🔍 Executing SQL:\n{safe_sql}")
        df, cols, rows, elapsed = execute_sql_safely(safe_sql)
        from utils.metrics import metrics
        metrics.log_sql_query(elapsed)

        # Filter by tech type if relevant
        if "type_tech" in df.columns:
            user_query_lower = ctx.query.lower()
            if any(w in user_query_lower for w in ["demand", "consumption", "loss", "export"]):
                demand_df = df[df["type_tech"].isin(DEMAND_TECH_TYPES)]
                if not demand_df.empty:
                    df = demand_df.copy()
                    log.info(f"⚙️ Showing DEMAND side only: {DEMAND_TECH_TYPES}")
            elif "transit" in user_query_lower:
                transit_df = df[df["type_tech"].isin(TRANSIT_TECH_TYPES)]
                if not transit_df.empty:
                    df = transit_df.copy()
                    log.info("⚙️ Showing TRANSIT data only.")
            else:
                supply_df = df[df["type_tech"].isin(SUPPLY_TECH_TYPES)]
                if not supply_df.empty:
                    df = supply_df.copy()
                    log.info(f"⚙️ Showing SUPPLY side only: {SUPPLY_TECH_TYPES}")

        ctx.df = df
        ctx.cols = list(df.columns)
        ctx.rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
        trace_detail(
            log,
            ctx,
            "stage_2_sql_execute",
            "sql_result",
            sql_hash=sql_query_hash(ctx.safe_sql or safe_sql),
            rows=len(ctx.rows),
            cols=len(ctx.cols),
            elapsed_ms=round(float(elapsed) * 1000.0, 2),
            sql_is_relevant=ctx.sql_is_relevant,
            skip_chart_due_to_relevance=ctx.skip_chart_due_to_relevance,
        )
        stamp_provenance(
            ctx,
            ctx.cols,
            ctx.rows,
            source="sql",
            query_hash=sql_query_hash(ctx.safe_sql or safe_sql),
        )

    except OperationalError as e:
        from utils.metrics import metrics
        metrics.log_error()
        trace_detail(
            log,
            ctx,
            "stage_2_sql_execute",
            "error",
            error_type="OperationalError",
            error=str(e),
            sql_hash=sql_query_hash(ctx.safe_sql or safe_sql),
        )
        log.error(f"⚠️ Database operational error: {e}")
        raise

    except DatabaseError as e:
        from utils.metrics import metrics
        metrics.log_error()
        msg = str(e)
        trace_detail(
            log,
            ctx,
            "stage_2_sql_execute",
            "error",
            error_type="DatabaseError",
            error=msg,
            sql_hash=sql_query_hash(ctx.safe_sql or safe_sql),
        )

        # Auto-pivot fix for hallucinated trade_derived_entities columns
        if "UndefinedColumn" in msg and "trade_derived_entities" in safe_sql:
            log.warning("🩹 Auto-pivoting trade_derived_entities: converting entity rows into share_* columns.")
            safe_sql = build_trade_share_cte(safe_sql)
            ctx.safe_sql = safe_sql
            df, cols, rows, _ = execute_sql_safely(safe_sql)
            ctx.df = df
            ctx.cols = list(df.columns)
            ctx.rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
            stamp_provenance(
                ctx,
                ctx.cols,
                ctx.rows,
                source="sql",
                query_hash=sql_query_hash(ctx.safe_sql),
            )

        elif "UndefinedColumn" in msg:
            # Column synonym auto-fix
            fixed = False
            for bad, good in COLUMN_SYNONYMS.items():
                if re.search(rf"\b{bad}\b", safe_sql, flags=re.IGNORECASE):
                    safe_sql = re.sub(rf"\b{bad}\b", good, safe_sql, flags=re.IGNORECASE)
                    log.warning(f"🔁 Auto-corrected column '{bad}' → '{good}' (retry)")
                    ctx.safe_sql = safe_sql
                    df, cols, rows, _ = execute_sql_safely(safe_sql)
                    ctx.df = df
                    ctx.cols = list(df.columns)
                    ctx.rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
                    stamp_provenance(
                        ctx,
                        ctx.cols,
                        ctx.rows,
                        source="sql",
                        query_hash=sql_query_hash(ctx.safe_sql),
                    )
                    fixed = True
                    break
            if not fixed:
                log.exception("SQL execution failed (UndefinedColumn)")
                raise
        else:
            log.exception("SQL execution failed (DatabaseError)")
            raise

    except SQLAlchemyError as e:
        from utils.metrics import metrics
        metrics.log_error()
        trace_detail(
            log,
            ctx,
            "stage_2_sql_execute",
            "error",
            error_type="SQLAlchemyError",
            error=str(e),
            sql_hash=sql_query_hash(ctx.safe_sql or safe_sql),
        )
        log.exception("SQLAlchemy error occurred")
        raise

    except Exception as e:
        from utils.metrics import metrics
        metrics.log_error()
        trace_detail(
            log,
            ctx,
            "stage_2_sql_execute",
            "error",
            error_type=type(e).__name__,
            error=str(e),
            sql_hash=sql_query_hash(ctx.safe_sql or safe_sql),
        )
        log.exception("Unexpected error during SQL execution")
        raise

    return ctx
