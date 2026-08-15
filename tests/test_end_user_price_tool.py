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
        "distribution": 0.0812,
        "supply": 0.1104,
        "transmission": 0.0067,
        "final_price_net": 0.1983,
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

    assert "final_price_net" in cols
    assert {"distribution", "supply", "transmission"} <= set(cols)
    assert "final_price" in captured["sql"]
    # Only the requested supplier/category is bound.
    assert captured["params"]["supplier_0_0"] == "telmico"
    assert captured["params"]["cat_id_0_0"] == "220/380|hh|cat2"


def test_vat_is_added_only_when_requested(monkeypatch):
    """value/final_price are NET of VAT; 18% is levied on top."""
    import agent.tools.end_user_price_tools as tool_module

    monkeypatch.setattr(tool_module, "run_text_query", _stub_rows())

    _, cols_net, _ = tool_module.get_end_user_prices(include_vat=False)
    assert "total_gross" not in cols_net

    df, cols_gross, _ = tool_module.get_end_user_prices(include_vat=True)
    assert {"vat", "total_gross"} <= set(cols_gross)
    assert round(float(df["total_gross"].iloc[0]), 4) == round(0.1983 * 1.18, 4)


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

    for component in ("distribution", "supply", "transmission", "final_price_net"):
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


def test_select_position_parameters_are_explicitly_cast(monkeypatch):
    """psycopg 3 binds server-side, so a bare parameter in the SELECT list has
    no inferable type and Postgres raises "could not determine data type of
    parameter". No other tool in this codebase binds a parameter in SELECT
    position, so there is no precedent protecting this one.
    """
    import re

    import agent.tools.end_user_price_tools as tool_module

    captured = {}

    def fake_run(sql, params=None):
        captured["sql"] = sql
        return _stub_rows()(sql, params)

    monkeypatch.setattr(tool_module, "run_text_query", fake_run)
    tool_module.get_end_user_prices(include_wholesale_benchmark=True)

    # Every ":param AS alias" in the SELECT list must carry a cast. The
    # negative lookbehind skips the "::text" cast itself, whose second colon
    # would otherwise read as a parameter named "text".
    uncast = re.findall(r"(?<!:):(\w+)\s+AS\s", captured["sql"])
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

    assert "final_price_net" in cols
    assert "total_gross" in cols, "include_vat was resolved but not honoured"
