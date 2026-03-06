"""
Typed retrieval tools for price queries.
"""
from typing import List, Optional

from config import MAX_ROWS
from .common import normalize_date, normalize_limit, run_text_query
from .types import ToolResult


ALLOWED_METRICS = {"balancing", "deregulated", "guaranteed_capacity", "exchange_rate"}
ALLOWED_CURRENCIES = {"gel", "usd", "both"}
ALLOWED_GRANULARITY = {"monthly", "yearly"}


def _resolve_columns(metric: str, currency: str) -> List[str]:
    if metric == "exchange_rate":
        return ["xrate"]

    metric_map = {
        "balancing": {"gel": "p_bal_gel", "usd": "p_bal_usd"},
        "deregulated": {"gel": "p_dereg_gel", "usd": "p_dereg_usd"},
        "guaranteed_capacity": {"gel": "p_gcap_gel", "usd": "p_gcap_usd"},
    }

    if currency == "both":
        return [metric_map[metric]["gel"], metric_map[metric]["usd"]]
    return [metric_map[metric][currency]]


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

    if granularity == "yearly":
        select_cols = ", ".join([f"AVG({c}) AS {c}" for c in cols])
        sql = f"""
SELECT
    EXTRACT(YEAR FROM date)::int AS year,
    {select_cols}
FROM price_with_usd
WHERE (:start_date IS NULL OR date >= :start_date)
  AND (:end_date IS NULL OR date <= :end_date)
GROUP BY 1
ORDER BY 1
LIMIT :limit
""".strip()
    else:
        select_cols = ", ".join(cols)
        sql = f"""
SELECT
    date,
    {select_cols}
FROM price_with_usd
WHERE (:start_date IS NULL OR date >= :start_date)
  AND (:end_date IS NULL OR date <= :end_date)
ORDER BY date
LIMIT :limit
""".strip()

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
    }
    return run_text_query(sql, params)

