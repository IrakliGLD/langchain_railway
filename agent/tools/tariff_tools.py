"""
Typed retrieval tools for tariff queries.
"""
from typing import Iterable, List, Optional

from config import MAX_ROWS
from .common import normalize_date, normalize_limit, run_text_query
from .types import ToolResult


# Public aliases group one or more plant-level tariff series into stable buckets.
TARIFF_ENTITY_ALIASES = {
    "enguri": ['ltd "engurhesi"1'],
    "gardabani_tpp": ['ltd "gardabni thermal power plant"'],
    "old_tpp_group": ['ltd "mtkvari energy"', 'ltd "iec" (tbilresi)', 'ltd "g power" (capital turbines)'],
    "regulated_hpp": [
        'ltd "engurhesi"1',
        'jsc "energo-pro georgia generation"',
        'ltd "vardnilihesi"',
    ],
    "regulated_new_tpp": ['ltd "gardabni thermal power plant"'],
    "regulated_old_tpp": ['ltd "mtkvari energy"', 'ltd "iec" (tbilresi)', 'ltd "g power" (capital turbines)'],
}
DEFAULT_TARIFF_ENTITY_ALIASES = ["enguri", "gardabani_tpp", "old_tpp_group"]
ALLOWED_CURRENCIES = {"gel", "usd"}


# Validate aliases early so the SQL builder only works with known entity groups.
def _validate_entities(entities: Optional[Iterable[str]]) -> List[str]:
    if not entities:
        return list(DEFAULT_TARIFF_ENTITY_ALIASES)
    normalized: List[str] = []
    for raw in entities:
        value = str(raw).strip().lower()
        if value not in TARIFF_ENTITY_ALIASES:
            raise ValueError(f"Unsupported tariff entity alias: {value}")
        normalized.append(value)
    return list(dict.fromkeys(normalized))


def get_tariffs(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    entities: Optional[List[str]] = None,
    currency: str = "gel",
    limit: int = MAX_ROWS,
) -> ToolResult:
    """Fetch key tariff series in pivot-style columns by date."""
    currency = str(currency or "gel").lower()
    if currency not in ALLOWED_CURRENCIES:
        raise ValueError(f"Unsupported tariff currency: {currency}")

    selected = _validate_entities(entities)
    start_date = normalize_date(start_date)
    end_date = normalize_date(end_date)
    limit = normalize_limit(limit)
    tariff_col = f"tariff_{currency}"

    # Each selected alias expands into a correlated subquery against the per-date tariff table.
    select_parts = []
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
        "enguri_entity": TARIFF_ENTITY_ALIASES["enguri"][0],
        "gardabani_entity": TARIFF_ENTITY_ALIASES["gardabani_tpp"][0],
        "old_tpp_1": TARIFF_ENTITY_ALIASES["old_tpp_group"][0],
        "old_tpp_2": TARIFF_ENTITY_ALIASES["old_tpp_group"][1],
        "old_tpp_3": TARIFF_ENTITY_ALIASES["old_tpp_group"][2],
    }

    if "enguri" in selected:
        select_parts.append(
            f"""(SELECT t.{tariff_col}
     FROM tariff_with_usd t
     WHERE t.date = d.date AND t.entity = :enguri_entity
     LIMIT 1) AS enguri_tariff_{currency}"""
        )
    if "gardabani_tpp" in selected:
        select_parts.append(
            f"""(SELECT t.{tariff_col}
     FROM tariff_with_usd t
     WHERE t.date = d.date AND t.entity = :gardabani_entity
     LIMIT 1) AS gardabani_tpp_tariff_{currency}"""
        )
    if "old_tpp_group" in selected:
        select_parts.append(
            f"""(SELECT AVG(t.{tariff_col})
     FROM tariff_with_usd t
     WHERE t.date = d.date
       AND t.entity IN (:old_tpp_1, :old_tpp_2, :old_tpp_3)) AS grouped_old_tpp_tariff_{currency}"""
        )
    if "regulated_hpp" in selected:
        select_parts.append(
            f"""(SELECT AVG(t.{tariff_col})
     FROM tariff_with_usd t
     WHERE t.date = d.date
       AND (t.entity ILIKE '%engurhesi%'
            OR t.entity ILIKE '%energo-pro%'
            OR t.entity ILIKE '%vardnili%')) AS regulated_hpp_tariff_{currency}"""
        )
    if "regulated_new_tpp" in selected:
        select_parts.append(
            f"""(SELECT t.{tariff_col}
     FROM tariff_with_usd t
     WHERE t.date = d.date AND t.entity = :gardabani_entity
     LIMIT 1) AS regulated_new_tpp_tariff_{currency}"""
        )
    if "regulated_old_tpp" in selected:
        select_parts.append(
            f"""(SELECT AVG(t.{tariff_col})
     FROM tariff_with_usd t
     WHERE t.date = d.date
       AND t.entity IN (:old_tpp_1, :old_tpp_2, :old_tpp_3)) AS regulated_old_tpp_tariff_{currency}"""
        )

    if not select_parts:
        raise ValueError("No tariff entities selected")

    # The outer date spine keeps all requested tariff groups aligned on one timeline.
    select_clause = ",\n    ".join(select_parts)
    where_parts = []
    if start_date:
        where_parts.append("date >= :start_date")
    if end_date:
        where_parts.append("date <= :end_date")
    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    sql = f"""
WITH dates AS (
    SELECT DISTINCT date
    FROM tariff_with_usd
    {where_clause}
)
SELECT
    d.date,
    {select_clause}
FROM dates d
ORDER BY d.date
LIMIT :limit
""".strip()

    return run_text_query(sql, params)
