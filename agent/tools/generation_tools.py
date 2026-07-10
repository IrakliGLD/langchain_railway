"""
Typed retrieval tools for generation mix queries.
"""
from typing import Iterable, List, Optional

from sqlalchemy import bindparam, text

from config import MAX_ROWS
from context import (
    DEMAND_TECH_TYPES,
    GENERATION_TECH_TYPES,
    SUPPLY_TECH_TYPES,
    TRANSIT_TECH_TYPES,
)

from .common import get_sort_direction, normalize_date, normalize_limit, run_statement
from .types import ToolResult

# Supported request enums are derived from the configured technology groups.
ALLOWED_MODES = {"quantity", "share"}
ALLOWED_GRANULARITY = {"monthly", "yearly"}
# Share denominators: "side" partitions by market side (supply/demand/transit),
# "generation" restricts the universe to domestic generation techs so shares
# express the generation mix (hydro+thermal+wind+solar sum to 1).
ALLOWED_SHARE_BASES = {"side", "generation"}
ALLOWED_TYPES = (
    set(SUPPLY_TECH_TYPES)
    | set(DEMAND_TECH_TYPES)
    | set(TRANSIT_TECH_TYPES)
)


def _sql_literal_list(values: Iterable[str]) -> str:
    literals = []
    for value in values:
        escaped = str(value).replace("'", "''")
        literals.append(f"'{escaped}'")
    return ", ".join(literals)


_SUPPLY_TYPES_SQL = _sql_literal_list(SUPPLY_TECH_TYPES)
_DEMAND_TYPES_SQL = _sql_literal_list(DEMAND_TECH_TYPES)
_TRANSIT_TYPES_SQL = _sql_literal_list(TRANSIT_TECH_TYPES)
_GENERATION_TYPES_SQL = _sql_literal_list(GENERATION_TECH_TYPES)


# Validate requested technology filters before they reach SQL construction.
def _validate_types(types: Optional[Iterable[str]]) -> List[str]:
    if not types:
        return []
    normalized = []
    for raw in types:
        value = str(raw).strip().lower()
        if value not in ALLOWED_TYPES:
            raise ValueError(f"Unsupported technology type: {value}")
        normalized.append(value)
    return list(dict.fromkeys(normalized))


def get_generation_mix(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    types: Optional[List[str]] = None,
    mode: str = "quantity",
    granularity: str = "monthly",
    limit: int = MAX_ROWS,
    share_basis: str = "side",
) -> ToolResult:
    """Fetch generation/demand mix as quantities or shares."""
    mode = str(mode).lower()
    granularity = str(granularity).lower()
    share_basis = str(share_basis).lower()
    if mode not in ALLOWED_MODES:
        raise ValueError(f"Unsupported generation mode: {mode}")
    if granularity not in ALLOWED_GRANULARITY:
        raise ValueError(f"Unsupported generation granularity: {granularity}")
    if share_basis not in ALLOWED_SHARE_BASES:
        raise ValueError(f"Unsupported generation share_basis: {share_basis}")

    selected_types = _validate_types(types)
    start_date = normalize_date(start_date)
    end_date = normalize_date(end_date)
    limit = normalize_limit(limit)
    direction = get_sort_direction(start_date, end_date)

    # Switch between monthly date output and yearly rollups from the same source view.
    if granularity == "yearly":
        period_expr = "EXTRACT(YEAR FROM date)::int AS period"
        period_ref = "period"
    else:
        period_expr = "date AS period"
        period_ref = "period"

    date_where_parts = []
    params = {"limit": limit}

    if start_date:
        date_where_parts.append("date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        date_where_parts.append("date <= :end_date")
        params["end_date"] = end_date

    date_where_clause = ("WHERE " + " AND ".join(date_where_parts)) if date_where_parts else ""
    type_filter_clause = ""
    if selected_types:
        params["types"] = selected_types
        type_filter_clause = "WHERE type_tech IN :types"

    quantity_where_parts = list(date_where_parts)
    if selected_types:
        quantity_where_parts.append("type_tech IN :types")
    quantity_where_clause = ("WHERE " + " AND ".join(quantity_where_parts)) if quantity_where_parts else ""

    if mode == "share" and share_basis == "generation":
        # Generation-mix basis: the denominator is TOTAL DOMESTIC GENERATION,
        # so hydro/thermal/wind/solar shares sum to 1. Import/self-cons and
        # demand-side rows are out of scope by definition.
        generation_where_parts = list(date_where_parts)
        generation_where_parts.append(f"type_tech IN ({_GENERATION_TYPES_SQL})")
        generation_where_clause = "WHERE " + " AND ".join(generation_where_parts)
        sql = f"""
WITH base AS (
    -- Aggregate quantity first so shares are computed within each requested period.
    SELECT
        {period_expr},
        type_tech,
        SUM(quantity_tech) AS quantity_tech
    FROM tech_quantity_view
    {generation_where_clause}
    GROUP BY period, type_tech
),
with_shares AS (
    SELECT
        {period_ref},
        type_tech,
        quantity_tech,
        ROUND(
            quantity_tech / NULLIF(
                SUM(quantity_tech) OVER (PARTITION BY {period_ref}),
                0
            ),
            4
        ) AS share_tech
    FROM base
)
SELECT
    {period_ref},
    type_tech,
    quantity_tech,
    share_tech
FROM with_shares
{type_filter_clause}
ORDER BY {period_ref} {direction}, type_tech
LIMIT :limit
""".strip()
    elif mode == "share":
        sql = f"""
WITH base AS (
    -- Aggregate quantity first so shares are computed within each requested period.
    SELECT
        {period_expr},
        type_tech,
        SUM(quantity_tech) AS quantity_tech
    FROM tech_quantity_view
    {date_where_clause}
    GROUP BY period, type_tech
),
classified AS (
    SELECT
        {period_ref},
        type_tech,
        quantity_tech,
        CASE
            WHEN type_tech IN ({_SUPPLY_TYPES_SQL}) THEN 'supply'
            WHEN type_tech IN ({_DEMAND_TYPES_SQL}) THEN 'demand'
            WHEN type_tech IN ({_TRANSIT_TYPES_SQL}) THEN 'transit'
            ELSE 'other'
        END AS tech_side
    FROM base
),
with_shares AS (
    SELECT
        {period_ref},
        type_tech,
        quantity_tech,
        ROUND(
            quantity_tech / NULLIF(
                SUM(quantity_tech) OVER (PARTITION BY {period_ref}, tech_side),
                0
            ),
            4
        ) AS share_tech
    FROM classified
)
SELECT
    {period_ref},
    type_tech,
    quantity_tech,
    share_tech
FROM with_shares
{type_filter_clause}
ORDER BY {period_ref} {direction}, type_tech
LIMIT :limit
""".strip()
    else:
        # Quantity mode returns the direct grouped totals without an extra share layer.
        sql = f"""
SELECT
    {period_expr},
    type_tech,
    SUM(quantity_tech) AS quantity_tech
FROM tech_quantity_view
{quantity_where_clause}
GROUP BY period, type_tech
ORDER BY period {direction}, type_tech
LIMIT :limit
""".strip()

    statement = text(sql)
    if selected_types:
        statement = statement.bindparams(bindparam("types", expanding=True))

    return run_statement(statement, params)
