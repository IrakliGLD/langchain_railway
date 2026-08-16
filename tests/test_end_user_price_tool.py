"""Contract for the end-user price tool.

The eight categories are ported from the ``company_mapping`` /
``category_mapping`` CTEs of ``public.demand_tariff_mv``. Rows 6 and 8 carry a
real irregularity: the supply component is filed under ``level_2_cat = 'other'``
while the matching distribution component has a blank one. Matching
``level_2_cat`` uniformly across all three components silently drops the
distribution row and yields an incomplete price.
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


def _stub_rows(**overrides):
    """A one-row result shaped like the tool's own SELECT."""
    row = {
        "date": "2026-06-01",
        "supplier": "telmico",
        "category": "220/380|hh|cat2",
        "category_label": "Household cat 2, 101-301 kWh (220/380)",
        "distribution_tariff_gel_kwh": 0.0812,
        "supply_tariff_gel_kwh": 0.1104,
        "transmission_tariff_gel_kwh": 0.0067,
        "final_price_net_gel_kwh": 0.1983,
    }
    row.update(overrides)

    def _run(sql, params=None):
        df = pd.DataFrame([row])
        return df, list(df.columns), [tuple(r) for r in df.itertuples(index=False, name=None)]

    return _run


def test_eight_categories_exist_for_both_suppliers():
    from agent.tools.end_user_price_tools import END_USER_CATEGORIES, SUPPLIER_TO_DISTRIBUTOR

    assert len(END_USER_CATEGORIES) == 8
    assert SUPPLIER_TO_DISTRIBUTOR == {"telmico": "telasi", "eps": "epg"}


def test_commercial_medium_and_high_voltage_use_a_blank_distribution_subclass():
    """Categories 6 and 8. Getting this wrong yields an incomplete price."""
    from agent.tools.end_user_price_tools import CATEGORY_BY_ID

    for category_id in ("3.3-6-10|com|other", "35-110|com|other"):
        category = CATEGORY_BY_ID[category_id]
        assert category.supply_level_2 == "other"
        assert category.distribution_level_2 == ""


def test_household_categories_match_on_all_three_components():
    from agent.tools.end_user_price_tools import CATEGORY_BY_ID

    category = CATEGORY_BY_ID["220/380|hh|cat2"]

    assert category.supply_activity == "universal"
    assert category.supply_level_2 == "cat2"
    assert category.distribution_level_2 == "cat2"


def test_commercial_other_categories_use_the_public_supply_activity():
    """`public` serves public-service commercial; `universal` serves households
    and small commercial. Swapping them selects a different tariff."""
    from agent.tools.end_user_price_tools import CATEGORY_BY_ID

    assert CATEGORY_BY_ID["220/380|com|other"].supply_activity == "public"
    assert CATEGORY_BY_ID["220/380|com|small"].supply_activity == "universal"


def test_transmission_row_is_national_and_classless():
    from agent.tools.end_user_price_tools import TRANSMISSION_ROW

    assert TRANSMISSION_ROW["company"] == "gse"
    # The MV's activity VALUE, not a result-frame column name.
    assert TRANSMISSION_ROW["activity"] == "transmission"
    assert TRANSMISSION_ROW["volate"] == ""
    assert TRANSMISSION_ROW["level_1_cat"] == ""
    assert TRANSMISSION_ROW["level_2_cat"] == ""


def test_tool_returns_components_and_the_published_net_total(monkeypatch):
    """The tool returns the breakdown AND the published final_price, so an
    answer never has to compute a total that exists in no row."""
    import agent.tools.end_user_price_tools as tool_module

    captured = {}

    def fake_run(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return _stub_rows()(sql, params)

    monkeypatch.setattr(tool_module, "run_text_query", fake_run)

    df, cols, rows = tool_module.get_end_user_prices(
        supplier="telmico", category="220/380|hh|cat2"
    )

    assert "final_price_net_gel_kwh" in cols
    assert {"distribution_tariff_gel_kwh", "supply_tariff_gel_kwh", "transmission_tariff_gel_kwh"} <= set(cols)
    assert "final_price" in captured["sql"]
    # Only the requested supplier/category is bound.
    assert captured["params"]["supplier_0_0"] == "telmico"
    assert captured["params"]["cat_id_0_0"] == "220/380|hh|cat2"


def test_vat_is_added_only_when_requested(monkeypatch):
    """value/final_price are NET of VAT; 18% is levied on top."""
    import agent.tools.end_user_price_tools as tool_module

    monkeypatch.setattr(tool_module, "run_text_query", _stub_rows())

    _, cols_net, _ = tool_module.get_end_user_prices(include_vat=False)
    assert "total_gross_gel_kwh" not in cols_net

    df, cols_gross, _ = tool_module.get_end_user_prices(include_vat=True)
    assert {"vat_gel_kwh", "total_gross_gel_kwh"} <= set(cols_gross)
    assert round(float(df["total_gross_gel_kwh"].iloc[0]), 4) == round(0.1983 * 1.18, 4)


def test_wholesale_benchmark_adds_the_capacity_charge_and_converts_down(monkeypatch):
    """Benchmark = (p_bal_gel + p_gcap_gel) / 1000, in GEL/kWh.

    The supply tariff already bundles the guaranteed capacity fee, so the
    charge goes on the WHOLESALE side; and prices are GEL/MWh, so they are
    divided by 1000 rather than the tariff being multiplied by it.
    """
    import agent.tools.end_user_price_tools as tool_module

    captured = {}

    def fake_run(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return _stub_rows()(sql, params)

    monkeypatch.setattr(tool_module, "run_text_query", fake_run)

    tool_module.get_end_user_prices(include_wholesale_benchmark=True)

    assert "p_gcap_gel" in captured["sql"]
    assert "p_bal_gel" in captured["sql"]
    # The unit conversion is inlined, not bound: it is a physical constant and
    # one fewer untyped parameter for the server to infer.
    assert "/ 1000.0" in captured["sql"]
    assert "* 1000" not in captured["sql"], "tariffs must never be scaled up"


def test_a_partial_stack_is_excluded(monkeypatch):
    """A row missing any component is dropped rather than reported partial."""
    import agent.tools.end_user_price_tools as tool_module

    captured = {}

    def fake_run(sql, params=None):
        captured["sql"] = sql
        return _stub_rows()(sql, params)

    monkeypatch.setattr(tool_module, "run_text_query", fake_run)
    tool_module.get_end_user_prices()

    for component in ("distribution_tariff_gel_kwh", "supply_tariff_gel_kwh", "transmission_tariff_gel_kwh", "final_price_net_gel_kwh"):
        assert f"s.{component} IS NOT NULL" in captured["sql"]


def test_query_is_bounded_by_the_last_published_final_price_month(monkeypatch):
    """demand_tariff_mv runs to 2030; complete prices stop at the last
    final_price month. MAX(date) on the view is a trap."""
    import agent.tools.end_user_price_tools as tool_module

    captured = {}

    def fake_run(sql, params=None):
        captured["sql"] = sql
        return _stub_rows()(sql, params)

    monkeypatch.setattr(tool_module, "run_text_query", fake_run)
    tool_module.get_end_user_prices()

    assert "MAX(date) FROM demand_tariff_mv WHERE activity = 'final_price'" in captured["sql"]


def test_unknown_category_is_rejected():
    from agent.tools.end_user_price_tools import get_end_user_prices

    with pytest.raises(ValueError, match="Unknown end-user category"):
        get_end_user_prices(category="not-a-category")


def test_a_distribution_company_is_rejected_as_a_supplier():
    """telasi and epg distribute; they do not supply. This is the most likely
    caller mistake, and guessing would silently answer the wrong question."""
    from agent.tools.end_user_price_tools import get_end_user_prices

    with pytest.raises(ValueError, match="Unknown supplier"):
        get_end_user_prices(supplier="telasi")


def test_tool_is_registered_and_executable():
    from agent.tools.registry import TOOL_REGISTRY, list_tools

    assert "get_end_user_prices" in TOOL_REGISTRY
    assert "get_end_user_prices" in list_tools()


def test_tool_name_enum_carries_the_tool():
    from contracts.question_analysis import ToolName

    assert ToolName.GET_END_USER_PRICES.value == "get_end_user_prices"


def test_unbounded_call_returns_the_most_recent_months(monkeypatch):
    """Matches the convention in agent/tools/common.get_sort_direction.

    With no date filters the LIMIT must capture the most recent records, not
    the oldest -- otherwise "how are end-user tariffs trending" answers from
    2021 and truncates before reaching today.
    """
    import agent.tools.end_user_price_tools as tool_module

    captured = {}

    def fake_run(sql, params=None):
        captured["sql"] = sql
        return _stub_rows()(sql, params)

    monkeypatch.setattr(tool_module, "run_text_query", fake_run)

    tool_module.get_end_user_prices()
    assert "ORDER BY s.date DESC" in captured["sql"]

    tool_module.get_end_user_prices(start_date="2024-01-01")
    assert "ORDER BY s.date ASC" in captured["sql"]


# ---------------------------------------------------------------------------
# Analyzer entity_scope -> (supplier, category) resolution
# ---------------------------------------------------------------------------

def _qa_with_scope(entity_scope: str, canonical_query: str = "test query in English"):
    """resolve_tool_params reads canonical_query_en in preference to raw_query,
    so any wording a test relies on must live there."""
    from contracts.question_analysis import QuestionAnalysis

    return QuestionAnalysis(
        **{
            "version": "question_analysis_v1",
            "raw_query": "test query",
            "canonical_query_en": canonical_query,
            "language": {"input_language": "en", "answer_language": "en"},
            "classification": {
                "query_type": "data_retrieval",
                "analysis_mode": "light",
                "intent": "test intent",
                "needs_clarification": False,
                "confidence": 0.9,
            },
            "routing": {
                "preferred_path": "tool",
                "needs_sql": False,
                "needs_knowledge": False,
                "prefer_tool": True,
                "needs_multi_tool": False,
                "evidence_roles": [],
            },
            "knowledge": {},
            "tooling": {
                "candidate_tools": [
                    {"name": "get_end_user_prices", "score": 0.9, "reason": "retail"}
                ]
            },
            "sql_hints": {},
            "entity_scope": entity_scope,
            "visualization": {
                "chart_requested_by_user": False,
                "chart_recommended": False,
                "chart_confidence": 0.0,
            },
            "analysis_requirements": {
                "needs_driver_analysis": False,
                "needs_correlation_context": False,
                "derived_metrics": [],
            },
        }
    )


@pytest.mark.parametrize(
    "entity_scope,expected_supplier,expected_category",
    [
        ("Telmico", "telmico", None),
        ("EP Georgia Supply", "eps", None),
        ("household_consumers", None, None),
        ("distribution_network_tariffs", None, None),
        ("", None, None),
        ("something_unrecognised", None, None),
    ],
)
def test_end_user_scope_never_resolves_to_a_wrong_category(
    entity_scope, expected_supplier, expected_category
):
    """An unrecognised scope must WIDEN, never guess.

    Returning a specific category for a scope we did not understand produces a
    confidently wrong answer -- the exact failure this whole change exists to
    remove. Widening is merely less helpful.
    """
    from agent.planner import resolve_tool_params

    params = resolve_tool_params(
        _qa_with_scope(entity_scope), "get_end_user_prices", "test query"
    )

    assert params is not None
    assert params.get("supplier") == expected_supplier
    assert params.get("category") == expected_category


def test_household_band_wording_resolves_to_its_category():
    """A named consumption band is specific enough to resolve safely."""
    from agent.planner import resolve_tool_params

    params = resolve_tool_params(
        _qa_with_scope("household cat2"), "get_end_user_prices", "test query"
    )

    assert params["category"] == "220/380|hh|cat2"


def test_vat_wording_in_the_query_requests_the_gross_column():
    from agent.planner import resolve_tool_params

    params = resolve_tool_params(
        _qa_with_scope("", canonical_query="what do households pay including VAT?"),
        "get_end_user_prices",
        "what do households pay including VAT?",
    )

    assert params["include_vat"] is True


def test_wholesale_wording_requests_the_benchmark_column():
    from agent.planner import resolve_tool_params

    params = resolve_tool_params(
        _qa_with_scope(
            "", canonical_query="how does the end-user price compare with the wholesale price?"
        ),
        "get_end_user_prices",
        "test query",
    )

    assert params["include_wholesale_benchmark"] is True


def _capture_sql(monkeypatch, **kwargs):
    """Run the tool against a stub and hand back the SQL and params it built."""
    import agent.tools.end_user_price_tools as tool_module

    captured = {}

    def fake_run(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params or {}
        return _stub_rows()(sql, params)

    monkeypatch.setattr(tool_module, "run_text_query", fake_run)
    tool_module.get_end_user_prices(**kwargs)
    return captured


def test_no_value_column_is_classified_as_a_summable_quantity(monkeypatch):
    """Every value column is GEL/kWh -- a per-unit tariff, never a volume.

    ``is_intensive_metric`` reads column names, and "supply" is in its
    EXTENSIVE token list because elsewhere it means supplied volume. A column
    named bare ``supply`` therefore gets a ``sum=`` in stats_hint: a per-kWh
    tariff added up across months, which is exactly the meaningless figure the
    aggregate rules exist to suppress. The unit suffix is what prevents it.
    """
    import agent.tools.end_user_price_tools as tool_module
    from analysis.stats import is_intensive_metric

    monkeypatch.setattr(tool_module, "run_text_query", _stub_rows())
    _, cols, _ = tool_module.get_end_user_prices(include_vat=True)
    value_columns = [c for c in cols if c.endswith("_gel_kwh")]

    assert value_columns, f"no unit-carrying value columns in {cols}"
    not_intensive = [c for c in value_columns if not is_intensive_metric(c)]
    assert not not_intensive, (
        "these per-kWh columns would be SUMMED across months in stats_hint: "
        f"{not_intensive}"
    )


def test_every_parameter_actually_binds_through_sqlalchemy(monkeypatch):
    """The SQL must survive ``text()``, not merely look well-formed.

    SQLAlchemy's bind regex carries a negative lookahead for ':', so
    ``:supplier::text`` is NOT recognised as a parameter -- the literal text
    ``:supplier`` is shipped to Postgres, which fails at the colon. That is a
    ProgrammingError at execution time and invisible to any test that only
    inspects the SQL string, which is how it reached production on
    2026-08-15: the tool raised, the pipeline fell back to LLM-authored SQL,
    and the answer averaged across suppliers and categories.

    Use ``CAST(:name AS text)``: parsed by SQLAlchemy AND typed for psycopg 3's
    server-side binding.
    """
    import re

    from sqlalchemy import text
    from sqlalchemy.dialects import postgresql

    captured = _capture_sql(monkeypatch, include_wholesale_benchmark=True)

    compiled = str(text(captured["sql"]).compile(dialect=postgresql.psycopg.dialect()))

    # Checking the name set is not enough: a name used in both SELECT and WHERE
    # position registers as "bound" off the WHERE occurrence while its SELECT
    # occurrence stays literal. Only the compiled string proves every site was
    # substituted. Line comments are stripped first -- a ':name' written inside
    # one is prose, not a parameter reference.
    executable = re.sub(r"--[^\n]*", "", compiled)
    leftover = re.findall(r"(?<![:\w]):(\w+)", executable)
    assert not leftover, (
        "parameter references survived compilation as literal text and will "
        f"raise a Postgres syntax error: {sorted(set(leftover))}"
    )


def test_the_statement_can_actually_be_bound(monkeypatch):
    """The binding must round-trip in BOTH directions.

    Asserting only "every value passed is declared" misses the reverse and
    more dangerous case: a name DECLARED in the SQL that nothing passes a
    value for. SQLAlchemy then raises StatementError before reaching the
    server, so there is no SQLSTATE and the log says only
    "type=StatementError" -- which is exactly what happened in production on
    2026-08-15.

    The source was prose: SQLAlchemy scans the WHOLE string including
    comments, so a comment mentioning ':name::text' contributed two phantom
    parameters -- ':nam' (the regex backtracks when the greedy ':name' fails
    its lookahead) and ':name' from a second mention. A comment explaining
    the bind-parameter trap re-created the bind-parameter trap.

    construct_params is the check that matters: it is what the driver calls,
    and it raises on exactly this.
    """
    from sqlalchemy import text
    from sqlalchemy.dialects import postgresql

    for kwargs in (
        {},
        {"include_vat": True},
        {"include_wholesale_benchmark": True},
        {"supplier": "telmico", "category": "220/380|hh|cat2"},
        {"start_date": "2024-01-01", "end_date": "2024-12-31"},
    ):
        captured = _capture_sql(monkeypatch, **kwargs)
        statement = text(captured["sql"])
        declared = set(statement._bindparams)
        passed = set(captured["params"])

        assert declared == passed, (
            f"bind parameters do not round-trip for {kwargs}: "
            f"declared but never passed={sorted(declared - passed)}, "
            f"passed but not declared={sorted(passed - declared)}"
        )

        compiled = statement.compile(dialect=postgresql.psycopg.dialect())
        compiled.construct_params(captured["params"])


def test_the_generated_sql_parses_as_postgres(monkeypatch):
    """Syntax check without a database.

    The statement is assembled by string concatenation across sixteen UNION
    branches with conditional date fragments, so a future edit can produce
    something that binds cleanly and still will not parse. sqlglot is already
    a dependency; parsing costs nothing and fails loudly.
    """
    import sqlglot

    for kwargs in (
        {},
        {"include_wholesale_benchmark": True},
        {"start_date": "2024-01-01"},
        {"end_date": "2024-12-31"},
        {"start_date": "2024-01-01", "end_date": "2024-12-31"},
        {"supplier": "eps", "category": "35-110|com|other"},
    ):
        captured = _capture_sql(monkeypatch, **kwargs)
        sqlglot.parse_one(captured["sql"], dialect="postgres")


def test_no_emitted_sql_comment_mentions_a_bind_parameter(monkeypatch):
    """Keep the prose out of the payload.

    Any ':word' inside a SQL comment becomes a real bind parameter, so the
    explanation of why casts are written a certain way belongs in the Python
    docstring, never in the SQL the driver receives.
    """
    import re

    captured = _capture_sql(monkeypatch, include_wholesale_benchmark=True)

    offenders = [
        comment
        for comment in re.findall(r"--[^\n]*", captured["sql"])
        if re.search(r"(?<![:\w]):\w", comment)
    ]
    assert not offenders, f"SQL comments introduce phantom bind parameters: {offenders}"


def test_select_position_parameters_are_explicitly_cast(monkeypatch):
    """psycopg 3 binds server-side, so a bare parameter in the SELECT list has
    no inferable type and Postgres raises "could not determine data type of
    parameter". No other tool in this codebase binds a parameter in SELECT
    position, so there is no precedent protecting this one.
    """
    import re

    captured = _capture_sql(monkeypatch, include_wholesale_benchmark=True)

    # A parameter reaching an output alias directly carries no inferable type.
    # CAST(:name AS ...) is the only accepted form -- see the test above for
    # why the ``::`` shorthand cannot be used here.
    uncast = re.findall(r"(?<!:):(\w+)\s+AS\s+\w+\s*,?\s*$", captured["sql"], re.MULTILINE)
    assert not uncast, f"SELECT-position parameters without an explicit cast: {uncast}"


def test_resolved_params_dispatch_cleanly_through_the_registry(monkeypatch):
    """End-to-end contract: whatever resolve_tool_params emits must be callable.

    The planner and the tool are edited independently, so a param the planner
    adds and the tool does not accept would only surface at runtime as a
    TypeError deep in execute_tool.
    """
    import agent.tools.end_user_price_tools as tool_module
    from agent.planner import resolve_tool_params
    from agent.tools.registry import execute_tool
    from agent.tools.types import ToolInvocation

    monkeypatch.setattr(tool_module, "run_text_query", _stub_rows())

    params = resolve_tool_params(
        _qa_with_scope("Telmico", canonical_query="telmico household tariffs including vat"),
        "get_end_user_prices",
        "telmico household tariffs including vat",
    )

    df, cols, rows = execute_tool(ToolInvocation(name="get_end_user_prices", params=params))

    assert "final_price_net_gel_kwh" in cols
    assert "total_gross_gel_kwh" in cols, "include_vat was resolved but not honoured"


def test_the_price_is_also_given_per_mwh(monkeypatch):
    """Wholesale prices are GEL/MWh, so any comparison restates these upward.

    A model doing that multiplication itself produces numbers present in no
    row, and strict-numeric grounding strips them: on 2026-08-15 fourteen
    tokens (167 ... 303) were rejected, cutting a 3,433-character answer to
    415. They were exactly the final prices expressed per MWh.
    """
    import agent.tools.end_user_price_tools as tool_module

    monkeypatch.setattr(tool_module, "run_text_query", _stub_rows())
    df, cols, _ = tool_module.get_end_user_prices()

    assert "final_price_net_gel_mwh" in cols
    assert round(float(df["final_price_net_gel_mwh"].iloc[0]), 1) == 198.3


def test_a_consumer_class_narrows_without_picking_one_category():
    """"for non-household consumers" names five commercial categories."""
    from agent.tools.end_user_price_tools import _resolve_selection

    _, categories = _resolve_selection(None, None, "com")
    assert len(categories) == 4
    assert all(c.level_1_cat == "com" for c in categories)

    _, households = _resolve_selection(None, None, "hh")
    assert len(households) == 4
    assert all(c.level_1_cat == "hh" for c in households)


def test_an_unknown_consumer_class_is_rejected():
    import pytest as _pytest

    from agent.tools.end_user_price_tools import _resolve_selection

    with _pytest.raises(ValueError, match="Unknown consumer class"):
        _resolve_selection(None, None, "industrial")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("for non-household consumers", "com"),
        ("non household customers", "com"),
        ("for households", "hh"),
        ("commercial customers", "com"),
        ("end-user prices", None),
    ],
)
def test_consumer_class_resolution(text, expected):
    """"non-household" must not resolve to households by containing the word."""
    from agent.tools.end_user_price_tools import resolve_consumer_class

    assert resolve_consumer_class(text) == expected


def test_the_benchmark_carries_every_fee_the_supply_tariff_bundles(monkeypatch):
    """Like-for-like, or the spread is partly just a missing fee.

    The retail supply tariff bundles both the guaranteed capacity fee and the
    ESCO service fee. Whichever is left off the wholesale side shows up as
    apparent retail margin.
    """
    import agent.tools.end_user_price_tools as tool_module

    captured = _capture_sql(monkeypatch, include_wholesale_benchmark=True)
    sql = captured["sql"]

    assert "p.p_bal_gel + p.p_gcap_gel" in sql, "balancing + capacity fee missing"
    assert str(tool_module.ESCO_SERVICE_FEE_GEL_KWH) in sql, "ESCO fee missing"
    # Both the benchmark column and the spread must use the same expression;
    # a spread computed against a different benchmark than the one reported
    # would be internally inconsistent.
    benchmark_occurrences = sql.count(
        f"+ {tool_module.ESCO_SERVICE_FEE_GEL_KWH})"
    )
    assert benchmark_occurrences == 2, (
        f"benchmark expression should appear in both columns, found {benchmark_occurrences}"
    )


def test_the_esco_fee_is_a_named_constant_not_a_literal():
    """A magic number in SQL cannot be found when the fee is revised."""
    from agent.tools.end_user_price_tools import ESCO_SERVICE_FEE_GEL_KWH

    assert ESCO_SERVICE_FEE_GEL_KWH == 0.00019


def test_the_benchmark_is_arithmetically_what_it_claims():
    """Guard the actual sum, not just the presence of the terms."""
    from agent.tools.end_user_price_tools import (
        ESCO_SERVICE_FEE_GEL_KWH,
        KWH_PER_MWH,
    )

    p_bal, p_gcap = 120.0, 12.0   # GEL/MWh
    benchmark = (p_bal + p_gcap) / KWH_PER_MWH + ESCO_SERVICE_FEE_GEL_KWH
    assert round(benchmark, 6) == 0.13219

    supply_tariff = 0.1104        # GEL/kWh
    assert round(supply_tariff - benchmark, 6) == -0.02179
