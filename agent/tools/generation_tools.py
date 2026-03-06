"""
Typed retrieval tools for generation mix queries.
"""
from typing import Iterable, List, Optional

from sqlalchemy import bindparam, text

from config import MAX_ROWS
from context import TECH_TYPE_GROUPS
from .common import normalize_date, normalize_limit, run_statement
from .types import ToolResult


ALLOWED_MODES = {"quantity", "share"}
ALLOWED_GRANULARITY = {"monthly", "yearly"}
ALLOWED_TYPES = (
    set(TECH_TYPE_GROUPS["supply"].keys())
    | set(TECH_TYPE_GROUPS["demand"].keys())
    | {"transit", "self-cons"}
)


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

    if granularity == "yearly":
        period_expr = "EXTRACT(YEAR FROM date)::int AS period"
        period_ref = "period"
    else:
        period_expr = "date AS period"
        period_ref = "period"

    type_filter = ""
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
    }

    statement = None
    if selected_types:
        type_filter = "AND type_tech IN :types"
        params["types"] = selected_types

    if mode == "share":
        sql = f"""
WITH base AS (
    SELECT
        {period_expr},
        type_tech,
        SUM(quantity_tech) AS quantity_tech
    FROM tech_quantity_view
    WHERE (:start_date IS NULL OR date >= :start_date)
      AND (:end_date IS NULL OR date <= :end_date)
      {type_filter}
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
        sql = f"""
SELECT
    {period_expr},
    type_tech,
    SUM(quantity_tech) AS quantity_tech
FROM tech_quantity_view
WHERE (:start_date IS NULL OR date >= :start_date)
  AND (:end_date IS NULL OR date <= :end_date)
  {type_filter}
GROUP BY period, type_tech
ORDER BY period, type_tech
LIMIT :limit
""".strip()

    statement = text(sql)
    if selected_types:
        statement = statement.bindparams(bindparam("types", expanding=True))

    return run_statement(statement, params)

