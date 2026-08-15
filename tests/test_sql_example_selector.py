"""Wiring and correctness regressions for few-shot SQL example selection.

``get_relevant_examples`` picks example categories by keyword and then loads
them from a second dict.  A category present in one map and not the other is
silently dropped, which is how four categories in this module became
unreachable.  These tests make that class of drift loud.
"""

import ast
import pathlib
import re

import pytest

from knowledge.sql_example_selector import (
    END_USER_PRICE_EXAMPLES,
    PLANT_FLEET_EXAMPLES,
    get_relevant_examples,
)

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Categories added for the dashboard-shared views. Scoped deliberately: the
# pre-existing categories reference a phantom ``time_month`` column and are
# tracked separately, so asserting over all of them here would just pin a bug.
_NEW_CATEGORIES = {
    "end_user_price": END_USER_PRICE_EXAMPLES,
    "plant_fleet": PLANT_FLEET_EXAMPLES,
}


def _allowed_tables() -> set:
    """Parse STATIC_ALLOWED_TABLES without importing config (needs live env)."""
    tree = ast.parse((_ROOT / "config.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "STATIC_ALLOWED_TABLES"
        ):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("STATIC_ALLOWED_TABLES not found in config.py")


def _sql_blocks(examples: str, *, keep_comments: bool = False) -> str:
    """Return only the executable SQL, so prose is not scanned.

    Two sources of false positives are removed:

    * Query/Plan prose before each ``---SQL---`` marker. Otherwise "generation
      comes from plants commissioned after 2020" registers ``plants`` as a table.
    * ``--`` comments inside the SQL. A comment warning *against* a pattern
      would otherwise read as the example using it.
    """
    sql = "\n".join(chunk.split("EXAMPLE ")[0] for chunk in examples.split("---SQL---")[1:])
    if keep_comments:
        return sql
    return re.sub(r"--[^\n]*", "", sql)


def _selector_maps() -> tuple[set, set]:
    """Extract the two category dicts from the function source.

    They are locals inside ``get_relevant_examples``, so read them from the AST
    rather than executing the function.
    """
    source = (_ROOT / "knowledge" / "sql_example_selector.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"category_keywords", "category_examples_map"}
        ):
            keys = [key.value for key in node.value.keys]
            found[node.targets[0].id] = keys
    return set(found["category_keywords"]), set(found["category_examples_map"])


def test_every_detected_category_is_loadable():
    """A keyword match that has no examples entry is detected and then dropped."""
    keywords, examples_map = _selector_maps()

    assert keywords - examples_map == set(), (
        "Categories detected by keyword but absent from category_examples_map: "
        f"{sorted(keywords - examples_map)} -- they match and then load nothing"
    )


def test_every_loadable_category_is_reachable():
    """Examples wired for loading but with no keywords can never be selected."""
    keywords, examples_map = _selector_maps()

    assert examples_map - keywords == set(), (
        f"Categories loadable but unreachable: {sorted(examples_map - keywords)}"
    )


def test_selector_ordering_puts_specific_categories_before_generic_ones():
    """Dict order is priority -- ``matched[:max_categories]`` slices it.

    "distribution tariff" contains "tariff"; "capacity factor" contains
    "capacity".  If the generic category came first it would take the slot and
    the model would get examples pointing at the wrong view.
    """
    source = (_ROOT / "knowledge" / "sql_example_selector.py").read_text(encoding="utf-8")
    keyword_block = source.split("category_keywords = {")[1].split("}")[0]
    order = re.findall(r'^\s{8}"(\w+)":', keyword_block, re.MULTILINE)

    assert order.index("end_user_price") < order.index("tariff")
    assert order.index("plant_fleet") < order.index("generation")


@pytest.mark.parametrize(
    "query,expected",
    [
        ("what is the end-user electricity tariff", "11."),
        ("breakdown of the distribution tariff for households", "11."),
        ("compare the final price between suppliers", "11."),
        ("what is the capacity factor of wind plants", "12."),
        ("how many plants are in each size band", "12."),
        ("generation by commissioning cohort", "12."),
    ],
)
def test_new_categories_are_selected_for_their_queries(query, expected):
    assert expected in get_relevant_examples(query), (
        f"{query!r} did not load the expected example category"
    )


def test_generation_side_tariff_query_still_gets_generation_examples():
    """Regression: adding the retail category must not steal plain tariff queries."""
    examples = get_relevant_examples("show me Enguri and Gardabani tariff trends")

    assert "8." in examples, "generation-side tariff examples no longer load"


def test_new_examples_reference_only_allowlisted_tables():
    allowed = _allowed_tables()

    for name, examples in _NEW_CATEGORIES.items():
        sql = _sql_blocks(examples)
        # Strip EXTRACT(<part> FROM <col>) so its FROM is not read as a table.
        sql = re.sub(r"EXTRACT\s*\([^)]*\)", "", sql, flags=re.IGNORECASE)
        ctes = {match.lower() for match in re.findall(r"(?:WITH|,)\s+(\w+)\s+AS\s*\(", sql, re.I)}
        refs = {match.lower() for match in re.findall(r"(?:FROM|JOIN)\s+([a-zA-Z_]\w*)", sql, re.I)}

        unknown = sorted(refs - ctes - allowed)
        assert not unknown, f"{name} examples reference non-allowlisted tables: {unknown}"


def test_new_examples_do_not_use_the_phantom_time_month_column():
    """``time_month`` is not a column on any view; the real one is ``date``.

    Pre-existing categories still contain it (tracked separately). New examples
    must not add to the problem.
    """
    for name, examples in _NEW_CATEGORIES.items():
        assert "time_month" not in examples, f"{name} examples use the phantom time_month column"


def test_end_user_examples_teach_the_load_bearing_rules():
    """Each assertion maps to a way the generated SQL would otherwise be wrong."""
    # Blank dimensions are '' and not NULL.
    assert "level_1_cat = ''" in END_USER_PRICE_EXAMPLES
    # The usable horizon is the last final_price month, not MAX(date).
    assert "WHERE activity = 'final_price'" in END_USER_PRICE_EXAMPLES
    # Supplier -> distributor pairing must appear, or components cannot be joined.
    assert "telasi" in END_USER_PRICE_EXAMPLES and "telmico" in END_USER_PRICE_EXAMPLES


def test_plant_fleet_examples_teach_the_load_bearing_rules():
    # facility_count is a stock: the example must select a single month.
    assert "MAX(date) FROM by_capacity" in PLANT_FLEET_EXAMPLES
    # The ratio-vs-percent trap must be called out.
    assert "capacity_factor_percent" in PLANT_FLEET_EXAMPLES
    # segment is 'total' only in these views, so no example may filter on
    # 'balancing'. Checked against executable SQL: a comment warning against
    # the pattern is expected and must not trip this.
    assert "balancing" not in _sql_blocks(PLANT_FLEET_EXAMPLES)
    assert "balancing" in _sql_blocks(PLANT_FLEET_EXAMPLES, keep_comments=True), (
        "the comment warning against the balancing filter was removed"
    )
