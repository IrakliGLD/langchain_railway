"""Typed retrieval tool for regulated end-user (retail) electricity prices.

Reads ``public.demand_tariff_mv``. The category matrix below is ported from
that view's own ``company_mapping`` / ``category_mapping`` CTEs -- if the view
is ever redefined, these must move with it.

Deliberately distinct from ``get_tariffs`` (what a regulated *plant* is paid,
GEL/MWh) and ``get_prices`` (wholesale, GEL/MWh). A retail question answered
from the generation-side tool produces a fluent, fully-shipped, entirely wrong
answer -- see the 2026-08-15 production trace.
"""
import re
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

#: Full legal names. The view stores short codes, and an answer that says
#: "eps" and "telmico" is quoting database keys at a reader who asked about
#: companies. Emitted as a column so the answer and the chart legend can use
#: the real name without the model having to remember the mapping.
SUPPLIER_DISPLAY_NAMES: Dict[str, str] = {
    "telmico": "Telmico (Tbilisi Electricity Supply Company)",
    "eps": "EPS (EP Georgia Supply)",
}

#: Short forms for chart legends, where the full name would not fit.
SUPPLIER_SHORT_NAMES: Dict[str, str] = {"telmico": "Telmico", "eps": "EPS"}

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


#: Wording specific enough to pin one supply company.
#:
#: Distribution-company names resolve to the SUPPLIER they pair with, because
#: a question naming Telasi is asking about the Tbilisi stack and that stack's
#: supply component is Telmico's. Only the supply code is ever emitted --
#: ``_resolve_selection`` rejects "telasi" as a supplier by design, and this
#: map is what keeps that rejection from being reachable through the planner.
SUPPLIER_ALIASES: Dict[str, Tuple[str, ...]] = {
    "telmico": ("telmico", "tbilisi electricity supply", "telasi"),
    "eps": ("ep georgia supply", "epgeorgia supply", "eps", "energo-pro georgia", "epg"),
}

#: Consumption-band and consumer-class wording specific enough to pin a single
#: category. Anything vaguer is left unresolved: the tool then widens to all
#: eight rather than guessing one.
CATEGORY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "220/380|hh|cat1": ("cat1", "cat 1", "first band", "up to 101"),
    "220/380|hh|cat2": ("cat2", "cat 2", "second band", "101-301", "101–301"),
    "220/380|hh|cat3": ("cat3", "cat 3", "third band", "above 301", "over 301"),
    "220/380|com|small": ("small commercial", "small business"),
}

#: Voltage / class / band wording, resolved independently and composed into a
#: category id. A flat alias list cannot express "Telmico, 3.3-6-10 kV,
#: commercial (public supply)" -- which is the very example the clarification
#: offers the user, so answering it exactly as instructed used to loop forever.
#: Longer spellings first: "35-110" must win before a bare "35" would, and
#: "small commercial" before "commercial".
_VOLTAGE_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("3.3-6-10", ("3.3-6-10", "3.3–6–10", "3-6-10", "3,3-6-10", "6-10", "medium voltage")),
    ("35-110", ("35-110", "35–110", "35-100", "35 110", "high voltage")),
    ("220/380", ("220/380", "220-380", "220 380", "380 v", "low voltage")),
)
_CLASS_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("hh", ("household", "households", "residential", "domestic", "hh")),
    ("com", ("commercial", "business", "non-household", "enterprise", "com")),
)
_BAND_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("cat1", ("cat1", "cat 1", "up to 101", "under 101", "first band")),
    ("cat2", ("cat2", "cat 2", "101-301", "101–301", "second band")),
    ("cat3", ("cat3", "cat 3", "above 301", "over 301", "third band")),
    ("small", ("small commercial", "small business", "small")),
    ("other", ("public supply", "public service", "public", "other")),
)


def _alias_matches(haystack: str, alias: str) -> bool:
    """Substring match bounded by non-alphanumerics.

    Plain ``in`` matches "eps" inside unrelated words, and the previous guard
    against that -- padding the alias with spaces -- failed on "EPS, 220/380 V"
    because a comma is not a space, so a fully scoped reply was treated as
    unscoped and the clarification repeated. Boundaries handle punctuation
    without banning the '/', '-' and '.' the voltage codes need.
    """
    return re.search(
        rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", haystack
    ) is not None


def _first_alias_hit(haystack: str, table) -> Optional[str]:
    for code, aliases in table:
        if any(_alias_matches(haystack, alias) for alias in aliases):
            return code
    return None


def _compose_category(haystack: str) -> Optional[str]:
    """Build a category id from voltage/class/band wording, if it names one.

    Only returns ids that actually exist: a plausible-sounding combination the
    view does not publish (household at 35-110 kV) resolves to nothing rather
    than to a neighbouring category.
    """
    voltage = _first_alias_hit(haystack, _VOLTAGE_ALIASES)
    consumer_class = _first_alias_hit(haystack, _CLASS_ALIASES)
    if not (voltage and consumer_class):
        return None

    band = _first_alias_hit(haystack, _BAND_ALIASES)
    candidates = [band] if band else []
    # Households outside 220/380 carry a blank band, commercial defaults to
    # "other"; try the stated band first, then the shape the view actually uses.
    candidates += ["", "other"] if consumer_class == "hh" else ["other", ""]

    for candidate in candidates:
        if candidate is None:
            continue
        composed = category_id(voltage, consumer_class, candidate)
        if composed in CATEGORY_BY_ID:
            return composed
    return None


def scope_haystack(entity_scope: Optional[str], query: Optional[str]) -> str:
    """The text that scope is read from.

    Single authority: the planner fills tool params from this and the pipeline
    decides whether to ask for clarification from it. Reading different text in
    the two places is how a question the analyzer HAD scoped
    (``entity_scope='Telmico; small commercial at 220/380 V'``) was still asked
    to name its scope, on 2026-08-15, twice.
    """
    return f"{entity_scope or ''} {query or ''}".strip().lower()

#: Wording that asks for the retail price to be set against the wholesale side.
WHOLESALE_COMPARISON_MARKERS: Tuple[str, ...] = (
    "wholesale", "balancing", "market price", "compare with the market",
)


#: Components in tariff order: the stack a customer's bill is built from.
COMPONENT_COLUMNS: Tuple[str, ...] = (
    "transmission_tariff_gel_kwh",
    "distribution_tariff_gel_kwh",
    "supply_tariff_gel_kwh",
)

#: The published total, net of VAT, and its gross twin when VAT was requested.
NET_TOTAL_COLUMN = "final_price_net_gel_kwh"
GROSS_TOTAL_COLUMN = "total_gross_gel_kwh"

#: Columns that identify which series a row belongs to.
SERIES_COLUMNS: Tuple[str, ...] = ("supplier", "category")

#: Human-readable "Company — Category" key, used as the chart legend entry.
SERIES_LABEL_COLUMN = "series_label"


def is_retail_price_frame(columns) -> bool:
    """Whether a result frame came from ``get_end_user_prices``.

    Identified by shape rather than by provenance so it also holds for a frame
    rebuilt downstream. Requires the series keys AND the published total: a
    frame carrying components alone cannot be charted as a price stack.
    """
    # set(columns) directly: a pandas Index has no usable truth value, so the
    # customary ``columns or ()`` guard raises instead of defaulting.
    present = set(columns) if columns is not None else set()
    return (
        set(SERIES_COLUMNS) <= present
        and NET_TOTAL_COLUMN in present
        and set(COMPONENT_COLUMNS) <= present
    )


def headline_price_column(columns) -> str:
    """The column an answer should lead with: gross when VAT was requested."""
    present = set(columns) if columns is not None else set()
    return GROSS_TOTAL_COLUMN if GROSS_TOTAL_COLUMN in present else NET_TOTAL_COLUMN


def resolve_scope(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort ``(supplier, category)`` from free text.

    Single authority for retail scope resolution: the planner uses it to fill
    tool params, and the pipeline uses it to decide whether a comparison is
    pinned down enough to answer. Two readers of the same vocabulary must not
    disagree about whether a question named a category.
    """
    haystack = f" {(text or '').strip().lower()} "

    supplier = _first_alias_hit(haystack, tuple(SUPPLIER_ALIASES.items()))

    category = None
    for known_id, aliases in CATEGORY_ALIASES.items():
        if any(_alias_matches(haystack, alias) for alias in aliases):
            category = known_id
            break
    if category is None:
        category = _compose_category(haystack)

    return supplier, category


def resolve_consumer_class(text: str) -> Optional[str]:
    """'hh' / 'com' when the text names a consumer CLASS but not a category.

    "for non-household consumers" names four commercial categories at once.
    Returning a class lets the tool cover all four instead of the caller
    having to pick one, or widening to households the question excluded.
    """
    haystack = f" {(text or '').strip().lower()} "
    # Checked first: "non-household" contains "household".
    if _alias_matches(haystack, "non-household") or _alias_matches(haystack, "non household"):
        return "com"
    return _first_alias_hit(haystack, _CLASS_ALIASES)


def asks_for_wholesale_comparison(text: str) -> bool:
    """Whether the question sets the retail price against the wholesale side."""
    haystack = (text or "").lower()
    return any(marker in haystack for marker in WHOLESALE_COMPARISON_MARKERS)


def _resolve_selection(
    supplier: Optional[str],
    category: Optional[str],
    consumer_class: Optional[str] = None,
) -> Tuple[Tuple[str, ...], Tuple[EndUserCategory, ...]]:
    """Validate and widen the selection. Unknown input raises, never guesses.

    ``consumer_class`` narrows to households or to non-households without
    pinning a single category -- "compare this for non-household consumers"
    names a class of four commercial categories, not one of them, and the
    honest answer covers all four rather than picking one.
    """
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

    if consumer_class is not None:
        wanted = str(consumer_class).strip().lower()
        if wanted not in {"hh", "com"}:
            raise ValueError(
                f"Unknown consumer class: {consumer_class!r}. Expected 'hh' "
                "(households) or 'com' (non-household / commercial)."
            )
        narrowed = tuple(c for c in categories if c.level_1_cat == wanted)
        if not narrowed:
            raise ValueError(
                f"No end-user category matches consumer class {wanted!r} "
                f"within the selected category {category!r}."
            )
        categories = narrowed

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

    Two binding rules govern the SQL below, both learned the hard way:

    1. SELECT-position parameters are wrapped in ``CAST(:name AS text)``.
       psycopg 3 binds server-side, so a bare parameter there has no inferable
       type and Postgres raises "could not determine data type of parameter".
       The ``::`` shorthand does NOT work: SQLAlchemy's bind regex has a
       negative lookahead for a colon, so it would not recognise the parameter
       at all and the literal text would reach Postgres as a syntax error.

    2. No SQL comment may contain a colon followed by a word. SQLAlchemy scans
       the entire string, comments included, so such prose becomes a real bind
       parameter that nothing supplies a value for -- raising StatementError
       before the query ever reaches the server. This paragraph lives in the
       docstring rather than in the emitted SQL for exactly that reason.
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
        CAST(:supplier_{tag} AS text)   AS supplier,
        CAST(:cat_id_{tag} AS text)     AS category,
        CAST(:cat_label_{tag} AS text)  AS category_label,
        MAX(d.value) FILTER (
            WHERE d.company = :distributor_{tag} AND d.activity = 'distribution'
              AND d.volate = :volate_{tag} AND d.level_1_cat = :l1_{tag}
              AND d.level_2_cat = :dist_l2_{tag}
        ) AS distribution_tariff_gel_kwh,
        MAX(d.value) FILTER (
            WHERE d.company = :supplier_{tag} AND d.activity = :supply_activity_{tag}
              AND d.volate = :volate_{tag} AND d.level_1_cat = :l1_{tag}
              AND d.level_2_cat = :supply_l2_{tag}
        ) AS supply_tariff_gel_kwh,
        MAX(d.value) FILTER (
            WHERE d.company = 'gse' AND d.activity = 'transmission'
              AND d.volate = '' AND d.level_1_cat = '' AND d.level_2_cat = ''
        ) AS transmission_tariff_gel_kwh,
        MAX(d.value) FILTER (
            WHERE d.company = :supplier_{tag} AND d.activity = 'final_price'
              AND d.volate = :volate_{tag} AND d.level_1_cat = :l1_{tag}
              AND d.level_2_cat = :final_l2_{tag}
        ) AS final_price_net_gel_kwh
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
        # The spread is measured on the SUPPLY component alone, not on the
        # final price. Transmission and distribution are paid on the network
        # whichever way the energy is procured, so they cancel out of the
        # comparison; including them would inflate the apparent gap by the
        # whole network stack and answer a question nobody asked.
        benchmark_select = (
            f",\n    (p.p_bal_gel + p.p_gcap_gel) / {KWH_PER_MWH}"
            " AS wholesale_benchmark_gel_kwh"
            f",\n    s.supply_tariff_gel_kwh - (p.p_bal_gel + p.p_gcap_gel)"
            f" / {KWH_PER_MWH} AS supply_vs_wholesale_spread_gel_kwh"
        )
        benchmark_join = "\nLEFT JOIN price_with_usd p ON p.date = s.date"

    params["row_limit"] = limit
    return f"""
WITH stacked AS ({union}
)
SELECT
    s.date, s.supplier, s.category, s.category_label,
    s.distribution_tariff_gel_kwh, s.supply_tariff_gel_kwh,
    s.transmission_tariff_gel_kwh, s.final_price_net_gel_kwh{benchmark_select}
FROM stacked s{benchmark_join}
WHERE s.distribution_tariff_gel_kwh IS NOT NULL
  AND s.supply_tariff_gel_kwh IS NOT NULL
  AND s.transmission_tariff_gel_kwh IS NOT NULL
  AND s.final_price_net_gel_kwh IS NOT NULL
ORDER BY s.date {direction}, s.supplier, s.category
LIMIT :row_limit
"""


def get_end_user_prices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    supplier: Optional[str] = None,
    category: Optional[str] = None,
    consumer_class: Optional[str] = None,
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

    ``final_price_net_gel_kwh`` is NET of VAT.  With ``include_vat`` the gross
    figure is returned as well, so the answer can quote a real column rather
    than computing a number that appears in no row.

    Every value column carries its unit in its name. That is not decoration:
    the wholesale benchmark is GEL/MWh at source and these are GEL/kWh, and
    ``supply`` on its own is classified as a volume by ``is_intensive_metric``,
    which would license summing a per-kWh tariff across months.
    """
    if str(currency or "gel").lower() != "gel":
        raise ValueError(
            "get_end_user_prices serves GEL/kWh only; the view stores no USD tariff."
        )

    suppliers, categories = _resolve_selection(supplier, category, consumer_class)
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

    if not df.empty:
        df = df.copy()
        # The same price expressed per MWh. Wholesale prices are GEL/MWh, so
        # any comparison restates these upward -- and a model that does the
        # multiplication itself produces numbers in no row, which
        # strict-numeric grounding then strips. On 2026-08-15 fourteen tokens
        # (167 ... 303) were rejected this way, cutting a 3,433-character
        # answer to 415: they were exactly these values, per MWh.
        if NET_TOTAL_COLUMN in df.columns:
            df["final_price_net_gel_mwh"] = df[NET_TOTAL_COLUMN] * KWH_PER_MWH
        # The view stores short codes. An answer that says "eps" is quoting a
        # database key at someone who asked about a company, and a chart legend
        # showing one unlabelled line is worse still, so the readable names
        # travel with the data.
        df["supply_company"] = df["supplier"].map(
            lambda code: SUPPLIER_DISPLAY_NAMES.get(code, code)
        )
        df["series_label"] = [
            f"{SUPPLIER_SHORT_NAMES.get(supplier, supplier)} — {label}"
            for supplier, label in zip(df["supplier"], df["category_label"])
        ]
        cols = list(df.columns)
        rows = [tuple(r) for r in df.itertuples(index=False, name=None)]

    if include_vat and not df.empty and "final_price_net_gel_kwh" in df.columns:
        # Computed here rather than in SQL so the arithmetic is visible and
        # testable, and so the gross figure exists as a real column in the
        # frame -- otherwise the grounding gate strips it from the answer.
        df = df.copy()
        df["vat_gel_kwh"] = df["final_price_net_gel_kwh"] * VAT_RATE
        df["total_gross_gel_kwh"] = df["final_price_net_gel_kwh"] * (1 + VAT_RATE)
        cols = list(df.columns)
        rows = [tuple(r) for r in df.itertuples(index=False, name=None)]

    return df, cols, rows
