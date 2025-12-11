"""
Test Suite for LLM Response Quality Evaluation

This module provides a manual testing framework for evaluating:
1. SQL correctness (tables, columns, filters)
2. Answer quality (language, completeness, accuracy)
3. Chart appropriateness
4. Response format consistency

Usage:
    python evaluation/test_suite.py --run-all
    python evaluation/test_suite.py --category energy_security
    python evaluation/test_suite.py --test-case 1

Manual Review Process:
    1. Run test case
    2. Review generated SQL, answer, and chart
    3. Score each aspect (1-5)
    4. Add notes about issues
    5. Track patterns over time
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum

# =============================================================================
# TEST CASE DEFINITIONS
# =============================================================================

class Category(Enum):
    ENERGY_SECURITY = "energy_security"
    BALANCING_PRICE = "balancing_price"
    SUPPORT_SCHEMES = "support_schemes"
    STAKEHOLDERS = "stakeholders"
    DEMAND = "demand"
    GENERATION = "generation"
    OWNERSHIP = "ownership"
    TARIFFS = "tariffs"
    EXCHANGE_RATE = "exchange_rate"
    SEASONAL = "seasonal"


@dataclass
class ExpectedResult:
    """Expected results for validation"""
    # SQL expectations
    expected_tables: List[str]  # e.g., ["tech_quantity_view"]
    expected_columns: Optional[List[str]] = None  # e.g., ["time_month", "quantity", "type_tech"]
    forbidden_tables: Optional[List[str]] = None  # Tables that shouldn't be used
    should_have_group_by: Optional[bool] = None
    should_have_where_clause: Optional[bool] = None

    # Plan expectations
    expected_intent: Optional[str] = None  # e.g., "trend_analysis"
    expected_target: Optional[str] = None  # e.g., "demand"

    # Answer expectations
    language: str = "en"  # "en", "ka", "ru"
    should_mention: Optional[List[str]] = None  # Keywords/concepts that should appear
    should_not_mention: Optional[List[str]] = None  # Things to avoid
    should_have_seasonal_breakdown: Optional[bool] = None  # For price queries
    should_clarify_thermal_imports: Optional[bool] = None  # For energy security

    # Chart expectations
    should_have_chart: Optional[bool] = None
    expected_chart_type: Optional[str] = None  # "line", "bar", "stacked_bar"


@dataclass
class TestCase:
    """Single test case"""
    id: int
    category: Category
    query: str
    description: str
    expected: ExpectedResult
    notes: Optional[str] = None


# =============================================================================
# TEST CASES - ENERGY SECURITY
# =============================================================================

ENERGY_SECURITY_TESTS = [
    TestCase(
        id=1,
        category=Category.ENERGY_SECURITY,
        query="როგორ შეაფასებ საქართველოს ენერგეტიკულ უსაფრთხოებას?",
        description="Energy security assessment in Georgian",
        expected=ExpectedResult(
            expected_tables=["tech_quantity_view"],
            expected_intent="general",
            expected_target="energy_security",
            language="ka",
            should_mention=["იმპორტი", "ჰიდრო", "თერმული", "გაზი"],  # import, hydro, thermal, gas
            should_clarify_thermal_imports=True,  # CRITICAL: Must mention thermal uses imported gas
            should_have_seasonal_breakdown=True
        ),
        notes="Must clarify that thermal generation uses imported gas, not treat it as local"
    ),

    TestCase(
        id=2,
        category=Category.ENERGY_SECURITY,
        query="What is Georgia's import dependence in winter?",
        description="Winter import dependence",
        expected=ExpectedResult(
            expected_tables=["tech_quantity_view"],
            expected_columns=["type_tech", "quantity"],
            should_have_where_clause=True,  # Filter for winter months
            language="en",
            should_mention=["import", "thermal", "gas", "winter"],
            should_clarify_thermal_imports=True
        ),
        notes="Should explain both direct import and gas imports for thermal"
    ),

    TestCase(
        id=3,
        category=Category.ENERGY_SECURITY,
        query="Can Georgia achieve energy independence?",
        description="Energy independence potential",
        expected=ExpectedResult(
            expected_tables=["tech_quantity_view"],
            language="en",
            should_mention=["hydro", "wind", "solar", "renewable"],
            should_not_mention=["thermal will solve", "thermal is local"],
            should_clarify_thermal_imports=True
        ),
        notes="Must emphasize renewables, not thermal, for true independence"
    ),
]

# =============================================================================
# TEST CASES - BALANCING PRICE
# =============================================================================

BALANCING_PRICE_TESTS = [
    TestCase(
        id=11,
        category=Category.BALANCING_PRICE,
        query="რატომ გაიზარდა საბალანსო ფასი 2024 წელს?",
        description="Why did balancing price increase in 2024? (Georgian)",
        expected=ExpectedResult(
            expected_tables=["price_with_usd", "trade_derived_entities"],
            expected_intent="correlation",
            expected_target="balancing_price",
            language="ka",
            should_mention=["გაზის ფასი", "კურსი", "წილი"],  # gas price, xrate, share
            should_have_seasonal_breakdown=True,
            should_have_chart=True
        ),
        notes="Should analyze composition changes (share_import, share_renewable_ppa) and xrate"
    ),

    TestCase(
        id=12,
        category=Category.BALANCING_PRICE,
        query="Compare summer and winter balancing prices 2023",
        description="Seasonal price comparison",
        expected=ExpectedResult(
            expected_tables=["price_with_usd"],
            expected_intent="comparison",
            should_have_group_by=True,  # GROUP BY season
            language="en",
            should_have_seasonal_breakdown=True,  # MANDATORY
            should_mention=["summer", "winter", "GEL/MWh"],
            should_not_mention=["annual average"]  # Should NOT use annual average only
        ),
        notes="CRITICAL: Must show summer AND winter averages separately, not annual"
    ),

    TestCase(
        id=13,
        category=Category.BALANCING_PRICE,
        query="What drives balancing electricity price?",
        description="Price drivers explanation",
        expected=ExpectedResult(
            expected_tables=["trade_derived_entities", "price_with_usd"],
            expected_intent="general",
            language="en",
            should_mention=["composition", "shares", "xrate", "hydro", "import", "renewable PPA"],
            should_have_chart=False  # Conceptual question, maybe no chart needed
        ),
        notes="Should explain composition first, then xrate, then other factors"
    ),

    TestCase(
        id=14,
        category=Category.BALANCING_PRICE,
        query="Show balancing price trends 2020-2024",
        description="Long-term price trends",
        expected=ExpectedResult(
            expected_tables=["price_with_usd"],
            expected_intent="trend_analysis",
            expected_target="balancing_price",
            language="en",
            should_have_seasonal_breakdown=True,
            should_have_chart=True,
            expected_chart_type="line"
        ),
        notes="Should separate summer/winter trends and explain different drivers"
    ),
]

# =============================================================================
# TEST CASES - DEMAND
# =============================================================================

DEMAND_TESTS = [
    TestCase(
        id=21,
        category=Category.DEMAND,
        query="Show me total demand from 2020 to 2023",
        description="Simple demand trend query",
        expected=ExpectedResult(
            expected_tables=["tech_quantity_view"],
            expected_columns=["time_month", "quantity", "type_tech"],
            should_have_where_clause=True,  # Filter for demand types
            forbidden_tables=["trade_derived_entities"],  # Should NOT use trade table
            expected_intent="trend_analysis",
            expected_target="total_demand",
            language="en",
            should_have_chart=True
        ),
        notes="Should use tech_quantity_view, NOT trade_derived_entities (per TableSelectionGuidance)"
    ),

    TestCase(
        id=22,
        category=Category.DEMAND,
        query="როგორია მოთხოვნის ზრდის ტემპი?",
        description="Demand growth rate (Georgian)",
        expected=ExpectedResult(
            expected_tables=["tech_quantity_view"],
            expected_intent="trend_analysis",
            language="ka",
            should_mention=["ზრდა", "პროცენტი", "CAGR"],  # growth, percent, CAGR
            should_have_chart=True
        ),
        notes="Should use seasonal statistics if available, mention CAGR"
    ),

    TestCase(
        id=23,
        category=Category.DEMAND,
        query="What causes demand fluctuations?",
        description="Demand drivers",
        expected=ExpectedResult(
            expected_tables=["tech_quantity_view"],
            language="en",
            should_mention=["seasonal", "weather", "industrial", "consumption"],
            should_have_chart=False  # Conceptual
        )
    ),
]

# =============================================================================
# TEST CASES - TARIFFS
# =============================================================================

TARIFF_TESTS = [
    TestCase(
        id=31,
        category=Category.TARIFFS,
        query="შეადარე ჰიდრო და თერმული ტარიფები",
        description="Compare hydro vs thermal tariffs (Georgian)",
        expected=ExpectedResult(
            expected_tables=["tariff_with_usd"],
            expected_intent="comparison",
            expected_target="tariff_comparison_hydro_thermal",
            language="ka",
            should_mention=["ჰიდრო", "თერმული", "GEL/MWh"],
            should_have_chart=True
        ),
        notes="Should group hydro entities separately from thermal entities"
    ),

    TestCase(
        id=32,
        category=Category.TARIFFS,
        query="How many regulated entities are there?",
        description="Count of regulated entities",
        expected=ExpectedResult(
            expected_tables=["tariff_with_usd"],
            expected_intent="list",
            language="en",
            should_mention=["hydro", "thermal", "entities"],
            should_have_chart=False  # Simple count query
        )
    ),

    TestCase(
        id=33,
        category=Category.TARIFFS,
        query="Show Enguri tariff trends 2020-2024",
        description="Specific entity tariff trends",
        expected=ExpectedResult(
            expected_tables=["tariff_with_usd"],
            expected_columns=["tariff_gel", "tariff_usd"],
            should_have_where_clause=True,  # Filter for Enguri
            expected_intent="trend_analysis",
            language="en",
            should_mention=["Enguri", "GEL/MWh", "trend"],
            should_have_chart=True
        )
    ),
]

# =============================================================================
# TEST CASES - SUPPORT SCHEMES
# =============================================================================

SUPPORT_SCHEME_TESTS = [
    TestCase(
        id=41,
        category=Category.SUPPORT_SCHEMES,
        query="რა არის CfD სქემა?",
        description="CfD explanation (Georgian)",
        expected=ExpectedResult(
            expected_tables=[],  # Conceptual question, no SQL needed
            expected_intent="general",
            language="ka",
            should_mention=["აუქციონი", "განახლებადი", "ფიქსირებული ფასი"],  # auction, renewable, fixed price
            should_have_chart=False
        ),
        notes="Conceptual question, should skip SQL generation (Option 4)"
    ),

    TestCase(
        id=42,
        category=Category.SUPPORT_SCHEMES,
        query="Show renewable PPA share trends",
        description="Renewable PPA trends",
        expected=ExpectedResult(
            expected_tables=["trade_derived_entities"],
            expected_intent="trend_analysis",
            should_have_where_clause=True,  # segment = 'Balancing Electricity'
            language="en",
            should_mention=["renewable PPA", "share", "percent"],
            should_have_chart=True
        ),
        notes="Should filter for segment='Balancing Electricity'"
    ),
]

# =============================================================================
# COMBINED TEST SUITE
# =============================================================================

ALL_TEST_CASES = (
    ENERGY_SECURITY_TESTS +
    BALANCING_PRICE_TESTS +
    DEMAND_TESTS +
    TARIFF_TESTS +
    SUPPORT_SCHEME_TESTS
)

# =============================================================================
# SCORING RUBRIC
# =============================================================================

SCORING_RUBRIC = """
SCORING RUBRIC (1-5 for each aspect):

1. SQL CORRECTNESS (1-5):
   5 = Perfect: Uses correct tables, columns, filters, aggregations
   4 = Good: Minor issues (e.g., unnecessary column, could be optimized)
   3 = Acceptable: Works but suboptimal (e.g., wrong table but gets result)
   2 = Poor: Major issues (e.g., wrong aggregation, missing filters)
   1 = Wrong: Completely incorrect or fails to execute

2. ANSWER QUALITY (1-5):
   5 = Excellent: Clear, accurate, well-formatted, follows all rules
   4 = Good: Accurate but minor formatting issues
   3 = Acceptable: Correct but could be clearer
   2 = Poor: Inaccurate or missing key information
   1 = Wrong: Incorrect information or misleading

3. LANGUAGE MATCH (1-5):
   5 = Perfect: Correct language, natural phrasing
   4 = Good: Correct language, minor awkwardness
   3 = Acceptable: Correct language but unnatural
   2 = Poor: Mixed languages or very awkward
   1 = Wrong: Wrong language

4. FORMAT COMPLIANCE (1-5):
   5 = Perfect: Follows all formatting rules (numbers, units, structure)
   4 = Good: Follows most rules, minor lapses
   3 = Acceptable: Some formatting issues
   2 = Poor: Many formatting violations
   1 = Wrong: No adherence to format rules

5. CRITICAL RULES (Pass/Fail):
   - Energy security: Clarifies thermal uses imported gas? YES/NO
   - Price comparison: Shows summer AND winter separately? YES/NO
   - Column names: No raw database names in answer? YES/NO
   - Seasonality: Uses seasonal stats if available? YES/NO

OVERALL SCORE = Average of scores 1-4, adjusted for critical rule failures
"""

# =============================================================================
# MANUAL TEST EXECUTION HELPER
# =============================================================================

def print_test_case(test_case: TestCase):
    """Print test case details for manual review"""
    print("=" * 80)
    print(f"TEST CASE #{test_case.id}: {test_case.description}")
    print("=" * 80)
    print(f"Category: {test_case.category.value}")
    print(f"Query: {test_case.query}")
    print(f"Language: {test_case.expected.language}")
    print()
    print("EXPECTED:")
    print(f"  Tables: {test_case.expected.expected_tables}")
    if test_case.expected.expected_columns:
        print(f"  Columns: {test_case.expected.expected_columns}")
    if test_case.expected.forbidden_tables:
        print(f"  Forbidden tables: {test_case.expected.forbidden_tables}")
    if test_case.expected.expected_intent:
        print(f"  Intent: {test_case.expected.expected_intent}")
    if test_case.expected.should_mention:
        print(f"  Should mention: {test_case.expected.should_mention}")
    if test_case.expected.should_not_mention:
        print(f"  Should NOT mention: {test_case.expected.should_not_mention}")
    if test_case.expected.should_have_seasonal_breakdown:
        print(f"  CRITICAL: Must have seasonal breakdown (summer/winter)")
    if test_case.expected.should_clarify_thermal_imports:
        print(f"  CRITICAL: Must clarify thermal uses imported gas")
    print()
    if test_case.notes:
        print(f"NOTES: {test_case.notes}")
    print()
    print("Now run this query in your system and review:")
    print(f'  curl -X POST http://localhost:8000/ask -d \'{{"query": "{test_case.query}"}}\'')
    print()
    print(SCORING_RUBRIC)
    print()


def run_category_tests(category: Category):
    """Print all tests for a category"""
    tests = [tc for tc in ALL_TEST_CASES if tc.category == category]
    print(f"\n{'=' * 80}")
    print(f"CATEGORY: {category.value.upper()} ({len(tests)} tests)")
    print(f"{'=' * 80}\n")
    for test in tests:
        print_test_case(test)
        input("Press Enter for next test...")


def run_single_test(test_id: int):
    """Print single test case"""
    test = next((tc for tc in ALL_TEST_CASES if tc.id == test_id), None)
    if test:
        print_test_case(test)
    else:
        print(f"Test case #{test_id} not found")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python evaluation/test_suite.py --run-all")
        print("  python evaluation/test_suite.py --category energy_security")
        print("  python evaluation/test_suite.py --test 1")
        print()
        print("Available categories:")
        for cat in Category:
            count = len([tc for tc in ALL_TEST_CASES if tc.category == cat])
            print(f"  - {cat.value} ({count} tests)")
        print()
        print(f"Total test cases: {len(ALL_TEST_CASES)}")
        sys.exit(0)

    command = sys.argv[1]

    if command == "--run-all":
        print(f"\nRunning all {len(ALL_TEST_CASES)} test cases...\n")
        for test in ALL_TEST_CASES:
            print_test_case(test)
            input("Press Enter for next test...")

    elif command == "--category":
        if len(sys.argv) < 3:
            print("Error: Please specify category")
            sys.exit(1)
        category_name = sys.argv[2]
        try:
            category = Category(category_name)
            run_category_tests(category)
        except ValueError:
            print(f"Error: Unknown category '{category_name}'")
            print("Available:", [c.value for c in Category])

    elif command == "--test":
        if len(sys.argv) < 3:
            print("Error: Please specify test ID")
            sys.exit(1)
        test_id = int(sys.argv[2])
        run_single_test(test_id)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
