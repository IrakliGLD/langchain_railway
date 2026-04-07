"""
Typed retrieval tools for price queries.
"""
from typing import List, Optional

from config import MAX_ROWS
from .common import get_sort_direction, normalize_date, normalize_limit, run_text_query
from .types import ToolResult


ALLOWED_METRICS = {"balancing", "deregulated", "guaranteed_capacity", "exchange_rate"}
ALLOWED_CURRENCIES = {"gel", "usd", "both"}
ALLOWED_GRANULARITY = {"monthly", "yearly"}


# Map semantic price requests onto the exact storage columns the tool exposes.
def _resolve_columns(metric: str, currency: str) -> List[str]:
    if metric == "exchange_rate":
        return ["xrate"]

    metric_map = {
        "balancing": {"gel": "p_bal_gel", "usd": "p_bal_usd"},
        "deregulated": {"gel": "p_dereg_gel", "usd": "p_dereg_usd"},
        "guaranteed_capacity": {"gel": "p_gcap_gel", "usd": "p_gcap_usd"},
    }

    cols = []
    if currency == "both":
        cols.extend([metric_map[metric]["gel"], metric_map[metric]["usd"]])
    else:
        cols.append(metric_map[metric][currency])
        
    # Balancing-price explanations often need both currencies plus xrate in one fetch.
    if metric == "balancing":
        for implicit_col in ["p_bal_gel", "p_bal_usd", "xrate"]:
            if implicit_col not in cols:
                cols.append(implicit_col)
                
    return cols


def get_prices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    currency: str = "gel",
    metric: str = "balancing",
    granularity: str = "monthly",
    limit: int = MAX_ROWS,
) -> ToolResult:
    """Fetch price series from price_with_usd with strict enum-validated params."""
    metric = str(metric).lower()
    currency = str(currency).lower()
    granularity = str(granularity).lower()

    if metric not in ALLOWED_METRICS:
        raise ValueError(f"Unsupported price metric: {metric}")
    if currency not in ALLOWED_CURRENCIES:
        raise ValueError(f"Unsupported currency: {currency}")
    if granularity not in ALLOWED_GRANULARITY:
        raise ValueError(f"Unsupported granularity: {granularity}")

    cols = _resolve_columns(metric, currency)
    start_date = normalize_date(start_date)
    end_date = normalize_date(end_date)
    limit = normalize_limit(limit)
    direction = get_sort_direction(start_date, end_date)

    # Build WHERE clauses dynamically so PostgreSQL sees each bind param once.
    where_parts: list[str] = []
    params: dict = {"limit": limit}
    if start_date:
        where_parts.append("date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_parts.append("date <= :end_date")
        params["end_date"] = end_date
    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    if granularity == "yearly":
        # Yearly mode averages each selected price column over the calendar year.
        select_cols = ", ".join([f"AVG({c}) AS {c}" for c in cols])
        sql = f"""
SELECT
    EXTRACT(YEAR FROM date)::int AS year,
    {select_cols}
FROM price_with_usd
{where_clause}
GROUP BY 1
ORDER BY 1 {direction}
LIMIT :limit
""".strip()
    else:
        # Monthly mode returns the stored series directly so downstream analysis keeps fidelity.
        select_cols = ", ".join(cols)
        sql = f"""
SELECT
    date,
    {select_cols}
FROM price_with_usd
{where_clause}
ORDER BY date {direction}
LIMIT :limit
""".strip()

    return run_text_query(sql, params)
