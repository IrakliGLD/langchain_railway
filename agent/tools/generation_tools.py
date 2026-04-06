"""
Typed retrieval tools for generation mix queries.
"""
from typing import Iterable, List, Optional

from sqlalchemy import bindparam, text

from config import MAX_ROWS
from context import TECH_TYPE_GROUPS
from .common import normalize_date, normalize_limit, run_statement
from .types import ToolResult


# Supported request enums are derived from the configured technology groups.
ALLOWED_MODES = {"quantity", "share"}
ALLOWED_GRANULARITY = {"monthly", "yearly"}
ALLOWED_TYPES = (
    set(TECH_TYPE_GROUPS["supply"].keys())
    | set(TECH_TYPE_GROUPS["demand"].keys())
    | {"transit", "self-cons"}
)


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
) -> ToolResult:
    """Fetch generation/demand mix as quantities or shares."""
    mode = str(mode).lower()
    granularity = str(granularity).lower()
    if mode not in ALLOWED_MODES:
        raise ValueError(f"Unsupported generation mode: {mode}")
    if granularity not in ALLOWED_GRANULARITY:
        raise ValueError(f"Unsupported generation granularity: {granularity}")

    selected_types = _validate_types(types)
    start_date = normalize_date(start_date)
    end_date = normalize_date(end_date)
    limit = normalize_limit(limit)

    # Switch between monthly date output and yearly rollups from the same source view.
    if granularity == "yearly":
        period_expr = "EXTRACT(YEAR FROM date)::int AS period"
        period_ref = "period"
    else:
        period_expr = "date AS period"
        period_ref = "period"

    where_parts = []
    params = {"limit": limit}

    if start_date:
        where_parts.append("date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_parts.append("date <= :end_date")
        params["end_date"] = end_date
    if selected_types:
        where_parts.append("type_tech IN :types")
        params["types"] = selected_types

    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    if mode == "share":
        sql = f"""
WITH base AS (
    -- Aggregate quantity first so shares are computed within each requested period.
    SELECT
        {period_expr},
        type_tech,
        SUM(quantity_tech) AS quantity_tech
    FROM tech_quantity_view
    {where_clause}
    GROUP BY period, type_tech
)
SELECT
    {period_ref},
    type_tech,
    quantity_tech,
    ROUND(quantity_tech / NULLIF(SUM(quantity_tech) OVER (PARTITION BY {period_ref}), 0), 4) AS share_tech
FROM base
ORDER BY {period_ref}, type_tech
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
{where_clause}
GROUP BY period, type_tech
ORDER BY period, type_tech
LIMIT :limit
""".strip()

    statement = text(sql)
    if selected_types:
        statement = statement.bindparams(bindparam("types", expanding=True))

    return run_statement(statement, params)
