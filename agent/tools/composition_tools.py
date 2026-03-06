"""
Typed retrieval tools for balancing composition queries.
"""
from typing import Iterable, List, Optional

from config import MAX_ROWS
from .common import normalize_date, normalize_limit, run_text_query
from .types import ToolResult


ALLOWED_BALANCING_ENTITIES = (
    "import",
    "deregulated_hydro",
    "regulated_hpp",
    "regulated_new_tpp",
    "regulated_old_tpp",
    "renewable_ppa",
    "thermal_ppa",
)


def _validate_entities(entities: Optional[Iterable[str]]) -> List[str]:
    if not entities:
        return list(ALLOWED_BALANCING_ENTITIES)
    normalized = []
    allowed = set(ALLOWED_BALANCING_ENTITIES)
    for e in entities:
        value = str(e).strip().lower()
        if value not in allowed:
            raise ValueError(f"Unsupported balancing entity: {value}")
        normalized.append(value)
    # Keep deterministic output order
    return [e for e in ALLOWED_BALANCING_ENTITIES if e in set(normalized)]


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

    all_entities_sql = ", ".join([f"'{e}'" for e in ALLOWED_BALANCING_ENTITIES])
    select_expr = ",\n    ".join(
        [f"MAX(CASE WHEN entity='{e}' THEN share ELSE 0 END) AS share_{e}" for e in selected_entities]
    )

    sql = f"""
WITH base AS (
    SELECT
        time_month AS date,
        entity,
        ROUND(SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY time_month), 0), 4) AS share
    FROM trade_derived_entities
    WHERE LOWER(REPLACE(segment, ' ', '_')) = 'balancing'
      AND entity IN ({all_entities_sql})
      AND (:start_date IS NULL OR time_month >= :start_date)
      AND (:end_date IS NULL OR time_month <= :end_date)
    GROUP BY time_month, entity
)
SELECT
    date,
    'balancing'::text AS segment,
    {select_expr}
FROM base
GROUP BY date
ORDER BY date
LIMIT :limit
""".strip()

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
    }
    return run_text_query(sql, params)
