"""Typed retrieval tool for regulated end-user (retail) electricity prices.

Reads ``public.demand_tariff_mv``. The category matrix below is ported from
that view's own ``company_mapping`` / ``category_mapping`` CTEs -- if the view
is ever redefined, these must move with it.

Deliberately distinct from ``get_tariffs`` (what a regulated *plant* is paid,
GEL/MWh) and ``get_prices`` (wholesale, GEL/MWh). A retail question answered
from the generation-side tool produces a fluent, fully-shipped, entirely wrong
answer -- see the 2026-08-15 production trace.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from config import MAX_ROWS

from .common import get_sort_direction, normalize_date, normalize_limit, run_text_query
from .types import ToolResult


@dataclass(frozen=True)
class EndUserCategory:
    """One final-price category and where its two variable components live."""

    id: str
    label: str
    volate: str
    level_1_cat: str
    level_2_cat: str
    supply_activity: str
    supply_level_2: str
    distribution_level_2: str


def category_id(volate: str, level_1_cat: str, level_2_cat: str) -> str:
    return f"{volate}|{level_1_cat}|{level_2_cat}"


#: Retail supplier -> the distribution company on whose network it supplies.
#: They are different legal entities; never substitute one for the other.
SUPPLIER_TO_DISTRIBUTOR: Dict[str, str] = {"telmico": "telasi", "eps": "epg"}

#: Transmission applies to every category and carries no voltage or class.
TRANSMISSION_ROW = {
    "company": "gse",
    "activity": "transmission",
    "volate": "",
    "level_1_cat": "",
    "level_2_cat": "",
}

#: value/final_price are stored NET of VAT; VAT is levied on top.
VAT_RATE = 0.18

#: Wholesale prices are GEL/MWh, tariffs GEL/kWh. Normalise downward.
KWH_PER_MWH = 1000.0

END_USER_CATEGORIES: Tuple[EndUserCategory, ...] = (
    EndUserCategory(
        "220/380|com|other", "Commercial - other (220/380)",
        "220/380", "com", "other", "public", "other", "other",
    ),
    EndUserCategory(
        "220/380|com|small", "Commercial - small (220/380)",
        "220/380", "com", "small", "universal", "small", "small",
    ),
    EndUserCategory(
        "220/380|hh|cat1", "Household cat 1, <=101 kWh (220/380)",
        "220/380", "hh", "cat1", "universal", "cat1", "cat1",
    ),
    EndUserCategory(
        "220/380|hh|cat2", "Household cat 2, 101-301 kWh (220/380)",
        "220/380", "hh", "cat2", "universal", "cat2", "cat2",
    ),
    EndUserCategory(
        "220/380|hh|cat3", "Household cat 3, >301 kWh (220/380)",
        "220/380", "hh", "cat3", "universal", "cat3", "cat3",
    ),
    # Rows 6 and 8: the supply component is filed under level_2_cat 'other'
    # while the matching distribution component has a BLANK one. This asymmetry
    # is real -- normalising it away silently drops the distribution row.
    EndUserCategory(
        "3.3-6-10|com|other", "Commercial - other (3.3-6-10)",
        "3.3-6-10", "com", "other", "public", "other", "",
    ),
    EndUserCategory(
        "3.3-6-10|hh|", "Household (3.3-6-10)",
        "3.3-6-10", "hh", "", "universal", "", "",
    ),
    EndUserCategory(
        "35-110|com|other", "Commercial - other (35-110)",
        "35-110", "com", "other", "public", "other", "",
    ),
)

CATEGORY_BY_ID: Dict[str, EndUserCategory] = {c.id: c for c in END_USER_CATEGORIES}


def _resolve_selection(
    supplier: Optional[str],
    category: Optional[str],
) -> Tuple[Tuple[str, ...], Tuple[EndUserCategory, ...]]:
    """Validate and widen the selection. Unknown input raises, never guesses."""
    if supplier is None:
        suppliers = tuple(SUPPLIER_TO_DISTRIBUTOR)
    else:
        key = str(supplier).strip().lower()
        if key not in SUPPLIER_TO_DISTRIBUTOR:
            raise ValueError(
                f"Unknown supplier: {supplier!r}. "
                f"Expected one of {sorted(SUPPLIER_TO_DISTRIBUTOR)} -- note that "
                "telasi and epg are distribution companies, not suppliers."
            )
        suppliers = (key,)

    if category is None:
        categories = END_USER_CATEGORIES
    else:
        key = str(category).strip()
        if key not in CATEGORY_BY_ID:
            raise ValueError(
                f"Unknown end-user category: {category!r}. "
                f"Expected one of {sorted(CATEGORY_BY_ID)}"
            )
        categories = (CATEGORY_BY_ID[key],)

    return suppliers, categories


def _build_sql(
    suppliers: Tuple[str, ...],
    categories: Tuple[EndUserCategory, ...],
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    include_wholesale_benchmark: bool,
    limit: int,
    direction: str,
    params: Dict[str, object],
) -> str:
    """One pivot per (date, supplier, category), plus the published total.

    A category's three components are looked up with that category's OWN keys.
    Components are never mixed across categories.
    """
    branches = []
    for index, supplier in enumerate(suppliers):
        distributor = SUPPLIER_TO_DISTRIBUTOR[supplier]
        for position, category in enumerate(categories):
            tag = f"{index}_{position}"
            params[f"supplier_{tag}"] = supplier
            params[f"distributor_{tag}"] = distributor
            params[f"volate_{tag}"] = category.volate
            params[f"l1_{tag}"] = category.level_1_cat
            params[f"supply_l2_{tag}"] = category.supply_level_2
            params[f"dist_l2_{tag}"] = category.distribution_level_2
            params[f"final_l2_{tag}"] = category.level_2_cat
            params[f"supply_activity_{tag}"] = category.supply_activity
            params[f"cat_id_{tag}"] = category.id
            params[f"cat_label_{tag}"] = category.label
            branches.append(
                f"""
    SELECT
        d.date,
        -- Explicit casts: psycopg 3 binds server-side, so a bare parameter in
        -- the SELECT list has no inferable type and Postgres raises
        -- "could not determine data type of parameter".
        :supplier_{tag}::text   AS supplier,
        :cat_id_{tag}::text     AS category,
        :cat_label_{tag}::text  AS category_label,
        MAX(d.value) FILTER (
            WHERE d.company = :distributor_{tag} AND d.activity = 'distribution'
              AND d.volate = :volate_{tag} AND d.level_1_cat = :l1_{tag}
              AND d.level_2_cat = :dist_l2_{tag}
        ) AS distribution,
        MAX(d.value) FILTER (
            WHERE d.company = :supplier_{tag} AND d.activity = :supply_activity_{tag}
              AND d.volate = :volate_{tag} AND d.level_1_cat = :l1_{tag}
              AND d.level_2_cat = :supply_l2_{tag}
        ) AS supply,
        MAX(d.value) FILTER (
            WHERE d.company = 'gse' AND d.activity = 'transmission'
              AND d.volate = '' AND d.level_1_cat = '' AND d.level_2_cat = ''
        ) AS transmission,
        MAX(d.value) FILTER (
            WHERE d.company = :supplier_{tag} AND d.activity = 'final_price'
              AND d.volate = :volate_{tag} AND d.level_1_cat = :l1_{tag}
              AND d.level_2_cat = :final_l2_{tag}
        ) AS final_price_net
    FROM demand_tariff_mv d
    WHERE TRUE
      {"AND d.date >= :start_date" if start_date else ""}
      {"AND d.date <= :end_date" if end_date else ""}
      AND d.date <= (SELECT MAX(date) FROM demand_tariff_mv WHERE activity = 'final_price')
    GROUP BY d.date"""
            )

    union = "\n    UNION ALL".join(branches)
    benchmark_select = ""
    benchmark_join = ""
    if include_wholesale_benchmark:
        # The supply tariff already bundles the guaranteed capacity fee, so the
        # charge is added to the WHOLESALE side; the regulated figure then stays
        # equal to what is actually charged. Prices are GEL/MWh -> divide.
        # KWH_PER_MWH is a physical constant, not user input, so it is inlined
        # rather than bound -- one less untyped parameter for the server to
        # infer, and it reads as the unit conversion it is.
        benchmark_select = (
            f",\n    (p.p_bal_gel + p.p_gcap_gel) / {KWH_PER_MWH} AS wholesale_benchmark"
            f",\n    s.final_price_net - (p.p_bal_gel + p.p_gcap_gel) / {KWH_PER_MWH} AS spread"
        )
        benchmark_join = "\nLEFT JOIN price_with_usd p ON p.date = s.date"

    params["row_limit"] = limit
    return f"""
WITH stacked AS ({union}
)
SELECT
    s.date, s.supplier, s.category, s.category_label,
    s.distribution, s.supply, s.transmission, s.final_price_net{benchmark_select}
FROM stacked s{benchmark_join}
WHERE s.distribution IS NOT NULL
  AND s.supply IS NOT NULL
  AND s.transmission IS NOT NULL
  AND s.final_price_net IS NOT NULL
ORDER BY s.date {direction}, s.supplier, s.category
LIMIT :row_limit
"""


def get_end_user_prices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    supplier: Optional[str] = None,
    category: Optional[str] = None,
    include_vat: bool = False,
    include_wholesale_benchmark: bool = False,
    currency: str = "gel",
    limit: int = MAX_ROWS,
) -> ToolResult:
    """Regulated end-user electricity prices and their components, in GEL/kWh.

    Returns one row per (date, supplier, category) with the distribution,
    supply and transmission components plus the regulator's published
    ``final_price``.  A row is emitted only when all three components and the
    published total are present -- a partial stack is worse than an absent one,
    and this mirrors the view's own INNER JOIN.

    ``final_price_net`` is NET of VAT.  With ``include_vat`` the gross figure is
    returned as well, so the answer can quote a real column rather than
    computing a number that appears in no row.
    """
    if str(currency or "gel").lower() != "gel":
        raise ValueError(
            "get_end_user_prices serves GEL/kWh only; the view stores no USD tariff."
        )

    suppliers, categories = _resolve_selection(supplier, category)
    start_date = normalize_date(start_date)
    end_date = normalize_date(end_date)
    limit = normalize_limit(limit)

    params: Dict[str, object] = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    sql = _build_sql(
        suppliers,
        categories,
        start_date=start_date,
        end_date=end_date,
        include_wholesale_benchmark=include_wholesale_benchmark,
        limit=limit,
        # No date filters -> DESC so the LIMIT captures the most recent months.
        direction=get_sort_direction(start_date, end_date),
        params=params,
    )

    df, cols, rows = run_text_query(sql, params)

    if include_vat and not df.empty and "final_price_net" in df.columns:
        # Computed here rather than in SQL so the arithmetic is visible and
        # testable, and so the gross figure exists as a real column in the
        # frame -- otherwise the grounding gate strips it from the answer.
        df = df.copy()
        df["vat"] = df["final_price_net"] * VAT_RATE
        df["total_gross"] = df["final_price_net"] * (1 + VAT_RATE)
        cols = list(df.columns)
        rows = [tuple(r) for r in df.itertuples(index=False, name=None)]

    return df, cols, rows
