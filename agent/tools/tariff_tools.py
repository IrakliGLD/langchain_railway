"""
Typed retrieval tools for tariff queries.
"""
from datetime import date
from typing import Iterable, List, Optional

from config import MAX_ROWS
from .common import get_sort_direction, normalize_date, normalize_limit, run_text_query
from .types import ToolResult


ENGURI_ENTITY = "enguri hpp"
VARDNILI_ENTITY = "vardnili hpp"
DZEVRULA_ENTITY = "dzevrula hpp"
GUMATI_ENTITY = "gumati hpp"
SHAORI_ENTITY = "shaori hpp"
RIONI_ENTITY = "rioni hpp"
LAJANURI_ENTITY = "lajanuri hpp"
ZHINVALI_ENTITY = "zhinvali hpp"
VARTSIKHE_ENTITY = "vartsikhe hpp"
KHRAMHESI_I_ENTITY = "khramhesi I"
KHRAMHESI_II_ENTITY = "khramhesi II"
GARDABANI_ENTITY = "gardabani tpp"
GPOWER_ENTITY = "gpower tpp"
MKTVARI_ENTITY = "mktvari tpp"
TBILSRESI_ENTITY = "tbilsresi tpp"

_ENTITY_DEREGULATION_START = {
    DZEVRULA_ENTITY: "2026-05-01",
    GUMATI_ENTITY: "2024-05-01",
    SHAORI_ENTITY: "2021-01-01",
    RIONI_ENTITY: "2022-05-01",
    LAJANURI_ENTITY: "2027-01-01",
    KHRAMHESI_I_ENTITY: "2027-01-01",
    KHRAMHESI_II_ENTITY: "2027-01-01",
}

_SINGLE_TARIFF_ALIAS_COLUMNS = {
    "enguri_hpp": "enguri_hpp",
    "enguri": "enguri",
    "vardnili_hpp": "vardnili_hpp",
    "vardnili": "vardnili",
    "dzevrula_hpp": "dzevrula_hpp",
    "dzevruli_hpp": "dzevruli_hpp",
    "gumati_hpp": "gumati_hpp",
    "shaori_hpp": "shaori_hpp",
    "rioni_hpp": "rioni_hpp",
    "lajanuri_hpp": "lajanuri_hpp",
    "zhinvali_hpp": "zhinvali_hpp",
    "vartsikhe_hpp": "vartsikhe_hpp",
    "khramhesi_i": "khramhesi_i",
    "khramhesi_ii": "khramhesi_ii",
    "gardabani_tpp": "gardabani_tpp",
    "gpower_tpp": "gpower_tpp",
    "mktvari_tpp": "mktvari_tpp",
    "mtkvari_tpp": "mtkvari_tpp",
    "tbilsresi_tpp": "tbilsresi_tpp",
    "tbilresi_tpp": "tbilsresi_tpp",
}


# Public aliases group one or more plant-level tariff series into stable buckets.
TARIFF_ENTITY_ALIASES = {
    "enguri_hpp": [ENGURI_ENTITY],
    "enguri": [ENGURI_ENTITY],
    "vardnili_hpp": [VARDNILI_ENTITY],
    "vardnili": [VARDNILI_ENTITY],
    "dzevrula_hpp": [DZEVRULA_ENTITY],
    "dzevruli_hpp": [DZEVRULA_ENTITY],
    "gumati_hpp": [GUMATI_ENTITY],
    "shaori_hpp": [SHAORI_ENTITY],
    "rioni_hpp": [RIONI_ENTITY],
    "lajanuri_hpp": [LAJANURI_ENTITY],
    "zhinvali_hpp": [ZHINVALI_ENTITY],
    "vartsikhe_hpp": [VARTSIKHE_ENTITY],
    "khramhesi_i": [KHRAMHESI_I_ENTITY],
    "khramhesi_ii": [KHRAMHESI_II_ENTITY],
    "gardabani_tpp": [GARDABANI_ENTITY],
    "gpower_tpp": [GPOWER_ENTITY],
    "mktvari_tpp": [MKTVARI_ENTITY],
    "mtkvari_tpp": [MKTVARI_ENTITY],
    "tbilsresi_tpp": [TBILSRESI_ENTITY],
    "tbilresi_tpp": [TBILSRESI_ENTITY],
    "old_tpp_group": [MKTVARI_ENTITY, TBILSRESI_ENTITY, GPOWER_ENTITY],
    "regulated_hpp": [
        ENGURI_ENTITY,
        VARDNILI_ENTITY,
        DZEVRULA_ENTITY,
        GUMATI_ENTITY,
        SHAORI_ENTITY,
        RIONI_ENTITY,
        LAJANURI_ENTITY,
        ZHINVALI_ENTITY,
        VARTSIKHE_ENTITY,
        KHRAMHESI_I_ENTITY,
        KHRAMHESI_II_ENTITY,
    ],
    "regulated_new_tpp": [GARDABANI_ENTITY],
    "regulated_old_tpp": [MKTVARI_ENTITY, TBILSRESI_ENTITY, GPOWER_ENTITY],
    "regulated_plants": [
        ENGURI_ENTITY,
        VARDNILI_ENTITY,
        DZEVRULA_ENTITY,
        GUMATI_ENTITY,
        SHAORI_ENTITY,
        RIONI_ENTITY,
        LAJANURI_ENTITY,
        ZHINVALI_ENTITY,
        VARTSIKHE_ENTITY,
        KHRAMHESI_I_ENTITY,
        KHRAMHESI_II_ENTITY,
        GARDABANI_ENTITY,
        GPOWER_ENTITY,
        MKTVARI_ENTITY,
        TBILSRESI_ENTITY,
    ],
}
DEFAULT_TARIFF_ENTITY_ALIASES = [
    "enguri_hpp",
    "vardnili_hpp",
    "dzevrula_hpp",
    "lajanuri_hpp",
    "zhinvali_hpp",
    "vartsikhe_hpp",
    "khramhesi_i",
    "khramhesi_ii",
    "gardabani_tpp",
    "gpower_tpp",
    "mktvari_tpp",
    "tbilsresi_tpp",
]
ALLOWED_CURRENCIES = {"gel", "usd", "both"}


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


def resolve_tariff_alias_entities(alias: str, as_of: Optional[date | str] = None) -> List[str]:
    """Resolve alias to plant entities, optionally filtered by deregulation date."""
    entities = list(TARIFF_ENTITY_ALIASES.get(str(alias).strip().lower(), []))
    if not entities or as_of is None:
        return entities

    if isinstance(as_of, str):
        as_of_date = date.fromisoformat(as_of[:10])
    else:
        as_of_date = as_of

    active: List[str] = []
    for entity in entities:
        dereg_start = _ENTITY_DEREGULATION_START.get(entity)
        if not dereg_start:
            active.append(entity)
            continue
        if as_of_date < date.fromisoformat(dereg_start):
            active.append(entity)
    return active


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
    direction = get_sort_direction(start_date, end_date)
    requested_currencies = ("gel", "usd") if currency == "both" else (currency,)

    # Build bind-friendly predicates once so the SQL can scan the tariff view
    # once per request instead of issuing one correlated lookup per alias/date.
    def _bind_entities(param_prefix: str, entities: List[str]) -> List[str]:
        bind_names = []
        for index, entity in enumerate(entities, start=1):
            bind_name = f"{param_prefix}_{index}"
            params[bind_name] = entity
            bind_names.append(bind_name)
        return bind_names

    def _in_predicate(
        param_prefix: str,
        entities: List[str],
        *,
        entity_col: str = "t.entity",
    ) -> str:
        bind_names = _bind_entities(param_prefix, entities)
        if not bind_names:
            return "1 = 0"
        return f"{entity_col} IN ({', '.join(f':{bind_name}' for bind_name in bind_names)})"

    def _dated_entity_predicate(param_prefix: str, entities: List[str]) -> str:
        clauses = []
        for index, entity in enumerate(entities, start=1):
            entity_bind = f"{param_prefix}_{index}"
            params[entity_bind] = entity
            dereg_start = _ENTITY_DEREGULATION_START.get(entity)
            if dereg_start:
                dereg_bind = f"{entity_bind}_dereg_start"
                params[dereg_bind] = dereg_start
                clauses.append(f"(t.entity = :{entity_bind} AND t.date < :{dereg_bind})")
            else:
                clauses.append(f"t.entity = :{entity_bind}")
        return " OR ".join(clauses) if clauses else "1 = 0"

    select_parts = []
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
    }
    relevant_entities = list(
        dict.fromkeys(
            entity
            for alias in selected
            for entity in TARIFF_ENTITY_ALIASES[alias]
        )
    )

    for alias, column_alias in _SINGLE_TARIFF_ALIAS_COLUMNS.items():
        if alias not in selected:
            continue
        bind_name = f"{column_alias}_entity"
        params[bind_name] = TARIFF_ENTITY_ALIASES[alias][0]
        for selected_currency in requested_currencies:
            select_parts.append(
                f"""MAX(CASE
        WHEN t.entity = :{bind_name} THEN t.tariff_{selected_currency}
    END) AS {column_alias}_tariff_{selected_currency}"""
            )
    if "old_tpp_group" in selected:
        old_tpp_predicate = _in_predicate("old_tpp_entity", TARIFF_ENTITY_ALIASES["old_tpp_group"])
        for selected_currency in requested_currencies:
            select_parts.append(
                f"""AVG(CASE
        WHEN {old_tpp_predicate} THEN t.tariff_{selected_currency}
    END) AS grouped_old_tpp_tariff_{selected_currency}"""
            )
    if "regulated_hpp" in selected:
        regulated_hpp_predicate = _dated_entity_predicate("regulated_hpp_entity", TARIFF_ENTITY_ALIASES["regulated_hpp"])
        for selected_currency in requested_currencies:
            select_parts.append(
                f"""AVG(CASE
        WHEN {regulated_hpp_predicate} THEN t.tariff_{selected_currency}
    END) AS regulated_hpp_tariff_{selected_currency}"""
            )
    if "regulated_new_tpp" in selected:
        params["gardabani_entity"] = GARDABANI_ENTITY
        for selected_currency in requested_currencies:
            select_parts.append(
                f"""MAX(CASE
        WHEN t.entity = :gardabani_entity THEN t.tariff_{selected_currency}
    END) AS regulated_new_tpp_tariff_{selected_currency}"""
            )
    if "regulated_old_tpp" in selected:
        regulated_old_tpp_predicate = _in_predicate("regulated_old_tpp_entity", TARIFF_ENTITY_ALIASES["regulated_old_tpp"])
        for selected_currency in requested_currencies:
            select_parts.append(
                f"""AVG(CASE
        WHEN {regulated_old_tpp_predicate} THEN t.tariff_{selected_currency}
    END) AS regulated_old_tpp_tariff_{selected_currency}"""
            )
    if "regulated_plants" in selected:
        regulated_plants_predicate = _dated_entity_predicate("regulated_plants_entity", TARIFF_ENTITY_ALIASES["regulated_plants"])
        for selected_currency in requested_currencies:
            select_parts.append(
                f"""AVG(CASE
        WHEN {regulated_plants_predicate} THEN t.tariff_{selected_currency}
    END) AS regulated_plants_tariff_{selected_currency}"""
            )

    if not select_parts:
        raise ValueError("No tariff entities selected")

    # Limit the date spine first, then read only the entities needed for this request.
    select_clause = ",\n    ".join(select_parts)
    where_parts = []
    if start_date:
        where_parts.append("date >= :start_date")
    if end_date:
        where_parts.append("date <= :end_date")
    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    filtered_entity_predicate = _in_predicate("selected_entity", relevant_entities, entity_col="src.entity")
    projected_tariff_cols = ",\n        ".join(
        f"src.tariff_{selected_currency}"
        for selected_currency in requested_currencies
    )

    sql = f"""
WITH dates AS (
    SELECT DISTINCT date
    FROM tariff_with_usd
    {where_clause}
    ORDER BY date {direction}
    LIMIT :limit
),
filtered_tariffs AS (
    SELECT
        src.date,
        src.entity,
        {projected_tariff_cols}
    FROM tariff_with_usd src
    JOIN dates d ON d.date = src.date
    WHERE {filtered_entity_predicate}
)
SELECT
    d.date,
    {select_clause}
FROM dates d
LEFT JOIN filtered_tariffs t ON t.date = d.date
GROUP BY d.date
ORDER BY d.date {direction}
""".strip()

    return run_text_query(sql, params)
