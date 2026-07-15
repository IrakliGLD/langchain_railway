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
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError, OperationalError, SQLAlchemyError

from agent.aggregation import validate_aggregation_logic
from agent.provenance import clear_provenance, sql_query_hash, stamp_provenance
from core.query_executor import ENGINE, execute_sql_safely
from core.sql_generator import plan_validate_repair, sanitize_sql, simple_table_whitelist_check
from models import QueryContext
from utils.query_validation import validate_sql_relevance
from utils.trace_logging import trace_detail

log = logging.getLogger("Enai")


# ---------------------------------------------------------------------------
# SQL column synonym auto-fix (from config.py)
# ---------------------------------------------------------------------------
try:
    from config import COLUMN_SYNONYMS
except ImportError:
    COLUMN_SYNONYMS = {}

# ---------------------------------------------------------------------------
# Tech type classifications (from context.py)
# ---------------------------------------------------------------------------
try:
    from context import DEMAND_TECH_TYPES, SUPPLY_TECH_TYPES, TRANSIT_TECH_TYPES
except ImportError:
    SUPPLY_TECH_TYPES = ["hydro", "thermal", "wind", "solar", "import", "self-cons"]
    DEMAND_TECH_TYPES = ["abkhazeti", "supply-distribution", "direct customers", "losses", "export"]
    TRANSIT_TECH_TYPES = ["transit"]


# ---------------------------------------------------------------------------
# Constants (moved from main.py)
# ---------------------------------------------------------------------------

BALANCING_SEGMENT_NORMALIZER = "balancing"

# type_tech side-narrowing intent keywords — canonical lists live in the
# shared lexicon (A3, 2026-06-10); aliases kept for import compatibility.
from contracts.intent_lexicon import (  # noqa: E402
    DEMAND_SIDE_KEYWORDS as DEMAND_INTENT_KEYWORDS,
)
from contracts.intent_lexicon import (
    SUPPLY_SIDE_KEYWORDS as SUPPLY_INTENT_KEYWORDS,
)


def _resolve_type_tech_side(query: str):
    """Return (label, tech_types) when the query explicitly targets one market
    side, else None. Demand intent wins over transit, which wins over supply;
    no match means a total/ambiguous query that should keep all rows."""
    q = (query or "").lower()
    if any(w in q for w in DEMAND_INTENT_KEYWORDS):
        return "demand", DEMAND_TECH_TYPES
    if "transit" in q:
        return "transit", TRANSIT_TECH_TYPES
    if any(w in q for w in SUPPLY_INTENT_KEYWORDS):
        return "supply", SUPPLY_TECH_TYPES
    return None


def apply_type_tech_side_filter(df: pd.DataFrame, query: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Narrow a ``type_tech`` result to one market side only on explicit intent.

    Returns (possibly_filtered_df, applied_side_label_or_None). A total or
    ambiguous query returns the dataframe unchanged with ``None`` — it must NOT
    be silently reduced to the supply side. If the matched side filters to an
    empty frame, the original frame is kept (no usable narrowing).
    """
    if "type_tech" not in df.columns:
        return df, None
    side = _resolve_type_tech_side(query)
    if side is None:
        return df, None
    label, tech_types = side
    filtered = df[df["type_tech"].isin(tech_types)]
    if filtered.empty:
        return df, None
    return filtered.copy(), label

# ---------------------------------------------------------------------------
# Balancing share pivot — single spec (Q6, 2026-06-10).
#
# The standalone pivot SQL and the CTE injected by build_trade_share_cte used
# to hand-maintain the same entity→share_* mapping twice; both are now
# rendered from the spec below. The generated strings are byte-identical to
# the former literals (verified against git HEAD at migration time);
# tests/test_sql_executor_pivot.py pins the shape.
#
# NOTE the deliberate asymmetry: ``share_total_hpp`` exists only in the
# standalone pivot (scope="pivot") — the CTE never had it, and adding it
# would change the CTE column surface. Preserve scopes when editing.
# ---------------------------------------------------------------------------

_SHARE_ENTITY_COLUMNS = [
    ("import", "share_import"),
    ("deregulated_hydro", "share_deregulated_hydro"),
    ("regulated_hpp", "share_regulated_hpp"),
    ("regulated_new_tpp", "share_regulated_new_tpp"),
    ("regulated_old_tpp", "share_regulated_old_tpp"),
    ("renewable_ppa", "share_renewable_ppa"),
    ("thermal_ppa", "share_thermal_ppa"),
    ("CfD_scheme", "share_cfd_scheme"),
]

# (column_alias, component_entities, scope) — scope ∈ {"both", "pivot"}
_SHARE_COMPOSITE_COLUMNS = [
    ("share_all_ppa", ["renewable_ppa", "thermal_ppa"], "both"),
    ("share_all_renewables", ["regulated_hpp", "deregulated_hydro", "renewable_ppa", "CfD_scheme"], "both"),
    ("share_total_hpp", ["regulated_hpp", "deregulated_hydro"], "pivot"),
]

_SHARE_RATIO_EXPR = "ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4)"


def _render_balancing_pivot_sql() -> str:
    """Render the standalone share pivot from the spec (aggregate MAX form)."""
    def term(e: str) -> str:
        return f"MAX(CASE WHEN entity='{e}' THEN share ELSE 0 END)"

    entity_lines = [f"    {term(e)} AS {col}," for e, col in _SHARE_ENTITY_COLUMNS]
    composite_lines = []
    for col, components, _scope in _SHARE_COMPOSITE_COLUMNS:
        joined = " +\n        ".join(term(e) for e in components)
        composite_lines.append(f"    {joined} AS {col},")
    body = "\n".join(entity_lines + composite_lines).rstrip(",")
    return f"""
SELECT
    date,
    '{BALANCING_SEGMENT_NORMALIZER}' AS segment,
{body}
FROM (
    SELECT date, entity,
           {_SHARE_RATIO_EXPR} AS share
    FROM trade_derived_entities
    WHERE LOWER(REPLACE(segment, ' ', '_')) = '{BALANCING_SEGMENT_NORMALIZER}'
    GROUP BY date, entity
) sub
GROUP BY date
ORDER BY date DESC
LIMIT 120
""".strip()


# Deterministic share pivot used both for direct fallback queries and repair paths.
BALANCING_SHARE_PIVOT_SQL = _render_balancing_pivot_sql()


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


def _render_trade_share_cte_body(cte_name: str) -> str:
    """Render the windowed-share CTE from the same spec as the standalone pivot."""
    def term(e: str) -> str:
        return f"MAX(CASE WHEN entity='{e}' THEN {_SHARE_RATIO_EXPR} ELSE 0 END) OVER (PARTITION BY date)"

    entity_lines = [f"        {term(e)} AS {col}," for e, col in _SHARE_ENTITY_COLUMNS]
    composite_lines = []
    for col, components, scope in _SHARE_COMPOSITE_COLUMNS:
        if scope != "both":
            continue  # share_total_hpp is pivot-only; the CTE never exposed it.
        joined = " +\n         ".join(term(e) for e in components)
        composite_lines.append(f"        ({joined}) AS {col},")
    body = "\n".join(entity_lines + composite_lines).rstrip(",")
    return f"""WITH {cte_name} AS (
    -- Materialize both raw quantity and reusable share_* columns in one place.
    SELECT
        date,
        entity,
        SUM(quantity) AS quantity,
        {_SHARE_RATIO_EXPR} AS share,
{body}
    FROM trade_derived_entities
    WHERE LOWER(REPLACE(segment, ' ', '_')) = 'balancing'
    GROUP BY date, entity
)
"""


def build_trade_share_cte(original_sql: str) -> str:
    """Inject a balancing electricity share pivot as a CTE and alias original SQL to it."""
    cte_name = "tde"
    cte = _render_trade_share_cte_body(cte_name)
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

    # Fallback to the canonical deterministic panel when the executed SQL omitted share columns.
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
    # Fix A (2026-05-16): Q4 production trace b21b9ece showed the legacy
    # planner emitting SQL that fails security/whitelist validation (an
    # unwhitelisted table reference or a parse error), causing
    # ``sanitize_sql`` / ``simple_table_whitelist_check`` to raise
    # ``HTTPException(400)``.  Previously that 400 propagated uncaught
    # through ``validate_and_execute`` to FastAPI, returning HTTP 400 to
    # the client and triggering 15+ retries via cache.  This is wrong:
    # for queries where the SQL is auxiliary (path=knowledge, conceptual
    # answer available, or any case where a topic-relevance failure would
    # have gracefully fallen back), a validation failure should also
    # gracefully degrade — set ``skip_sql=True`` with a clear reason and
    # let the pipeline fall through to the conceptual-summary path
    # (matches the existing handling at line ~277 for relevance failures).
    # Security is not weakened: the SQL is still NOT executed; only the
    # error-response shape changes from 400 to a useful conceptual answer.
    try:
        sanitized = sanitize_sql(ctx.raw_sql.strip())
        simple_table_whitelist_check(sanitized)
        safe_sql = plan_validate_repair(sanitized)
    except HTTPException as exc:
        ctx.df = pd.DataFrame()
        ctx.rows = []
        ctx.cols = []
        clear_provenance(ctx)
        failure_reason = f"sql_validation_failed:{exc.detail}"
        ctx.skip_sql = True
        ctx.skip_sql_reason = failure_reason
        log.warning(
            "⚠️ SQL validation failed (HTTP %s); falling back to conceptual answer. detail=%s",
            exc.status_code,
            exc.detail,
        )
        trace_detail(
            log,
            ctx,
            "stage_2_sql_execute",
            "validation_failed",
            reason=failure_reason,
            raw_sql_preview=(ctx.raw_sql or "")[:300],
            status_code=exc.status_code,
        )
        return ctx

    # Validate aggregation logic
    is_valid_aggregation, validation_reason = validate_aggregation_logic(safe_sql, ctx.aggregation_intent)
    if not is_valid_aggregation:
        log.warning(f"⚠️ SQL doesn't match aggregation intent: {validation_reason}")
    else:
        log.info(f"✅ SQL validation passed: {validation_reason}")

    semantic_query = ctx.effective_query

    # Force pivot injection for balancing share queries. When Stage 0.2 locked
    # semantics, use the resolved query so raw follow-up wording cannot hijack
    # the fallback SQL path.
    if should_inject_balancing_pivot(semantic_query, safe_sql):
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
        semantic_query, safe_sql, ctx.plan
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
        log.info("Executing SQL. sql_hash=%s", sql_query_hash(safe_sql))
        df, cols, rows, elapsed = execute_sql_safely(safe_sql)
        from utils.metrics import metrics
        metrics.log_sql_query(elapsed)

        # Narrow mixed generation results to the side of the market implied by
        # the question — but only on explicit intent. Total/ambiguous queries
        # keep all rows (no silent supply-only default; audit L1).
        if "type_tech" in df.columns:
            df, applied_side = apply_type_tech_side_filter(df, ctx.query)
            if applied_side:
                log.info("⚙️ Showing %s side only (explicit intent)", applied_side.upper())
                trace_detail(
                    log,
                    ctx,
                    "stage_2_sql_execute",
                    "type_tech_side_filter",
                    side=applied_side,
                )

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
        log.error("Database operational error. error_class=%s", type(e).__name__)
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

        # Repair a common failure mode where SQL references share columns that do not exist yet.
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
            # Retry once with configured column synonyms before surfacing the DB
            # error. Apply *every* matching synonym in a single pass so a query
            # that references two or more renamed columns is repaired in one
            # retry instead of failing after only the first was corrected.
            applied: list[str] = []
            for bad, good in COLUMN_SYNONYMS.items():
                if re.search(rf"\b{bad}\b", safe_sql, flags=re.IGNORECASE):
                    safe_sql = re.sub(rf"\b{bad}\b", good, safe_sql, flags=re.IGNORECASE)
                    applied.append(f"{bad}→{good}")
            if applied:
                log.warning("🔁 Auto-corrected columns %s (retry)", ", ".join(applied))
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
            else:
                log.error("SQL execution failed. error_class=UndefinedColumn")
                raise
        else:
            log.error("SQL execution failed. error_class=%s", type(e).__name__)
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
        log.error("SQLAlchemy error occurred. error_class=%s", type(e).__name__)
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
        log.error("Unexpected SQL execution error. error_class=%s", type(e).__name__)
        raise

    return ctx
