"""
Typed retrieval tools for balancing composition queries.
"""
from typing import Iterable, List, Optional

from config import MAX_ROWS
from .common import get_sort_direction, normalize_date, normalize_limit, run_text_query
from .types import ToolResult


# Canonical balancing buckets accepted by the composition tool.
ALLOWED_BALANCING_ENTITIES = (
    "import",
    "deregulated_hydro",
    "regulated_hpp",
    "regulated_new_tpp",
    "regulated_old_tpp",
    "renewable_ppa",
    "thermal_ppa",
    "CfD_scheme",
)


# Normalize user-provided entity names while preserving the public output order.
def _validate_entities(entities: Optional[Iterable[str]]) -> List[str]:
    if not entities:
        return list(ALLOWED_BALANCING_ENTITIES)
    normalized = []
    allowed_lower = {e.lower(): e for e in ALLOWED_BALANCING_ENTITIES}
    for e in entities:
        value = str(e).strip().lower()
        if value not in allowed_lower:
            raise ValueError(f"Unsupported balancing entity: {value}")
        normalized.append(allowed_lower[value])
    # Keep deterministic output order
    seen = set(normalized)
    return [e for e in ALLOWED_BALANCING_ENTITIES if e in seen]


def get_balancing_composition(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    entities: Optional[List[str]] = None,
    limit: int = MAX_ROWS,
) -> ToolResult:
    """Fetch balancing composition shares in pivot format."""
    selected_entities = _validate_entities(entities)
    start_date = normalize_date(start_date)
    end_date = normalize_date(end_date)
    limit = normalize_limit(limit)
    direction = get_sort_direction(start_date, end_date)

    # Build the dynamic pivot only for the validated entity set requested.
    all_entities_sql = ", ".join([f"'{e}'" for e in ALLOWED_BALANCING_ENTITIES])
    select_expr = ",\n    ".join(
        [f"MAX(CASE WHEN entity='{e}' THEN share ELSE 0 END) AS share_{e}" for e in selected_entities]
    )

    # Apply the time filters once so the share calculation and outer pivot stay aligned.
    where_parts = [
        "LOWER(REPLACE(segment, ' ', '_')) = 'balancing'",
        f"entity IN ({all_entities_sql})"
    ]
    params = {"limit": limit}

    if start_date:
        where_parts.append("date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_parts.append("date <= :end_date")
        params["end_date"] = end_date
    
    where_clause = " AND ".join(where_parts)

    sql = f"""
WITH base AS (
    -- First calculate each entity's share of total balancing quantity per date.
    SELECT
        date,
        entity,
        ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY date), 0), 4) AS share
    FROM trade_derived_entities
    WHERE {where_clause}
    GROUP BY date, entity
)
SELECT
    date,
    'balancing'::text AS segment,
    {select_expr}
FROM base
GROUP BY date
ORDER BY date {direction}
LIMIT :limit
""".strip()

    return run_text_query(sql, params)
