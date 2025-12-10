# Energy Chatbot - Comprehensive Code Audit
**Date:** 2025-12-10
**Auditor:** AI Engineer Review
**Version:** v18.7 (Gemini Analyst)

---

## Executive Summary

This audit identifies **critical architectural and functional issues** in the energy chatbot that affect:
1. **Calculation Accuracy** - Incorrect SQL generation for aggregations
2. **Chart-Answer Mismatch** - Charts don't support what the answer says
3. **Code Maintainability** - 3,900-line monolithic file
4. **Documentation Overload** - 7 redundant markdown files (~106KB)

**Priority Issues:**
- 🔴 **CRITICAL:** SQL generation for total calculations is unreliable
- 🔴 **CRITICAL:** Chart and answer are generated separately, leading to inconsistencies
- 🟡 **HIGH:** Massive monolithic architecture (main.py 167KB)
- 🟡 **HIGH:** Excessive documentation files with redundant content

---

## 1. Documentation Issues

### Problem: 7 Markdown Files with Significant Overlap

| File | Size | Issue |
|------|------|-------|
| CODE_REVIEW_REPORT.md | 22 KB | Detailed code review - useful but could be condensed |
| BEST_PRACTICES_REVIEW.md | 39 KB | **TOO LARGE** - industry comparisons, mostly theoretical |
| PERFORMANCE_ANALYSIS.md | 8.4 KB | Bottleneck analysis - specific to Railway deployment |
| TIMEOUT_QUALITY_EVALUATION.md | 12.6 KB | Quality testing plan - implementation-specific |
| EVALUATION_QUICKSTART.md | 6.4 KB | **Duplicate** of EVALUATION_GUIDE content |
| EVALUATION_GUIDE.md | 9.4 KB | Test queries and criteria |
| EVALUATION_RAILWAY_GUIDE.md | 8.3 KB | Production evaluation - Railway-specific |

**Total:** ~106 KB of documentation (vs 809 KB total repository)

### Issues:
- **Redundancy:** 3 evaluation guides cover overlapping content
- **Outdated:** Performance analysis references Railway hobby plan issues
- **Organization:** Files scattered in root directory
- **Maintenance:** Code changes not reflected in 7 different files

### Recommendation:
**Consolidate to 3 files in `docs/` directory:**
1. `DEVELOPER_GUIDE.md` - Setup, architecture, key functions (consolidate CODE_REVIEW + BEST_PRACTICES)
2. `EVALUATION.md` - How to run tests, interpret results (consolidate 3 evaluation guides)
3. `CHANGELOG.md` - Version history and optimization notes (consolidate PERFORMANCE + TIMEOUT)

**Savings:** ~60% reduction in doc files, better organization

---

## 2. Model Calculation Issues 🔴 CRITICAL

### Problem 1: Unreliable SQL Generation for Aggregations

**Location:** `llm_generate_plan_and_sql()` (main.py:1829-1968)

**Issue:**
The system uses LLM to generate SQL queries, which is inherently unreliable for precise calculations like:
- **Total generation** = SUM(quantity_tech) for all technologies
- **Share calculations** = entity_quantity / total_quantity
- **Seasonal aggregations** = Different logic for SUM vs AVG

**Example Failure Scenario:**
```
User: "What was total generation in 2023?"

LLM might generate:
  SELECT type_tech, quantity_tech FROM tech_quantity_view WHERE EXTRACT(YEAR FROM date) = 2023
  ❌ WRONG - doesn't SUM across technologies

Should generate:
  SELECT SUM(quantity_tech) as total_generation FROM tech_quantity_view
  WHERE EXTRACT(YEAR FROM date) = 2023 AND type_tech IN ('hydro', 'thermal', 'wind', 'solar')
  ✅ CORRECT
```

**Root Cause:**
- Line 1844: "You are an analytical PostgreSQL generator" - too generic
- Few-shot examples focus on filtering, not aggregations
- No validation that SUM/GROUP BY logic is correct
- LLM doesn't understand the difference between:
  - "Total generation" (SUM across all techs)
  - "Generation by technology" (GROUP BY type_tech)

**Evidence from Code:**
```python
# main.py:1684-1803 - Few-shot SQL examples
# Only 1 example (Example 11) shows SUM aggregation for total generation
# Other 12 examples focus on filtering, joins, time ranges
# NO examples for:
#   - Multi-level aggregations (total → by entity → by tech)
#   - CASE WHEN for seasonal grouping with aggregation
#   - Share calculations (entity/total)
```

**Impact:**
- Users asking "total generation" get individual rows instead of sum
- "Share of hydro" queries might not divide by correct total
- Seasonal comparisons might use wrong aggregation (SUM vs AVG)

### Problem 2: Share Calculations Hardcoded, Not Queryable

**Location:** `build_balancing_correlation_df()` (main.py:677-752)

**Issue:**
Share calculations are computed in Python code, not available via SQL:
```python
# Line 717-727: Share calculations hardcoded
share_import = qty_import / total_qty
share_deregulated_hydro = qty_dereg_hydro / total_qty
# ... etc
```

**Problem:**
- LLM-generated SQL can't access these shares directly
- User asks "show me share of renewables" → LLM must generate complex SQL to replicate Python logic
- Risk of inconsistency between Python shares and SQL-generated shares

**Should Be:**
- Materialized view in database: `balancing_shares_mv`
- Or: SQL function that LLM can call
- Ensures single source of truth

### Problem 3: No Validation of Calculation Logic

**Location:** SQL execution (main.py:2409-2451)

**Issue:**
After LLM generates SQL, there's NO validation that:
- SUM is used when total is requested
- GROUP BY matches the aggregation intent
- Denominators for shares are correct
- Seasonal logic uses correct months (4-7 vs 8-12, 1-3)

**Example:**
```python
# main.py:2409 - SQL validation only checks table whitelist
validated_sql = validate_and_fix_sql(raw_sql)  # Only checks table names
# NO CHECK: Is this a SUM when total is requested?
# NO CHECK: Is GROUP BY correct for the aggregation?
```

### Recommendations:

#### Fix 1: Add Aggregation Intent Detection (HIGH PRIORITY)
```python
def detect_aggregation_intent(user_query: str) -> dict:
    """Detect if user wants total, average, breakdown, etc."""
    query_lower = user_query.lower()

    intent = {
        "needs_total": any(k in query_lower for k in ["total", "sum", "overall", "all", "სულ"]),
        "needs_average": any(k in query_lower for k in ["average", "mean", "საშუალო"]),
        "needs_breakdown": any(k in query_lower for k in ["by", "breakdown", "each", "per"]),
        "needs_share": any(k in query_lower for k in ["share", "percentage", "proportion", "წილი"]),
    }
    return intent
```

#### Fix 2: Validate SQL Matches Intent
```python
def validate_aggregation_logic(sql: str, intent: dict) -> tuple[bool, str]:
    """Ensure SQL matches aggregation intent."""
    sql_upper = sql.upper()

    if intent["needs_total"]:
        if "SUM(" not in sql_upper and "sum(" not in sql:
            return False, "Total requested but SQL doesn't use SUM"

    if intent["needs_average"]:
        if "AVG(" not in sql_upper and "avg(" not in sql:
            return False, "Average requested but SQL doesn't use AVG"

    if intent["needs_breakdown"] and intent["needs_total"]:
        if "GROUP BY" not in sql_upper:
            return False, "Breakdown requested but SQL doesn't use GROUP BY"

    return True, "OK"
```

#### Fix 3: Add Aggregation Examples to Few-Shot
```sql
-- Example: Total generation across all technologies
SELECT
  SUM(quantity_tech) * 1000 AS total_generation_mwh
FROM tech_quantity_view
WHERE EXTRACT(YEAR FROM date) = 2023
  AND type_tech IN ('hydro', 'thermal', 'wind', 'solar', 'import');

-- Example: Total generation BY technology
SELECT
  type_tech,
  SUM(quantity_tech) * 1000 AS total_generation_mwh
FROM tech_quantity_view
WHERE EXTRACT(YEAR FROM date) = 2023
GROUP BY type_tech
ORDER BY total_generation_mwh DESC;

-- Example: Share of each technology
WITH totals AS (
  SELECT
    type_tech,
    SUM(quantity_tech) AS tech_total
  FROM tech_quantity_view
  WHERE EXTRACT(YEAR FROM date) = 2023
  GROUP BY type_tech
),
grand_total AS (
  SELECT SUM(tech_total) AS overall_total FROM totals
)
SELECT
  t.type_tech,
  t.tech_total,
  gt.overall_total,
  ROUND(t.tech_total / gt.overall_total * 100, 2) AS share_percent
FROM totals t, grand_total gt
ORDER BY share_percent DESC;
```

---

## 3. Charting Issues 🔴 CRITICAL

### Problem 1: Chart and Answer Generated Separately

**Location:**
- Answer generation: `llm_summarize()` (main.py:2129-2320)
- Chart generation: Chart builder logic (main.py:3466-3700)

**Issue Flow:**
```
1. SQL executed → rows fetched
2. LLM generates answer from data preview
3. SEPARATE process generates chart from same data
4. No feedback loop between answer and chart
```

**Example Failure:**
```
User: "Show me balancing price trend and key drivers"

Answer (from LLM):
  "Balancing price increased from 45 to 78 GEL/MWh due to:
   - Import share rising from 15% to 35%
   - Exchange rate depreciation from 2.7 to 3.1
   - Thermal generation replacing cheap hydro"

Chart (from automatic logic):
  Line chart showing ONLY p_bal_gel over time
  ❌ Missing: import share, exchange rate, thermal vs hydro

Why? Chart logic selected "line" for time series price,
but didn't include the drivers mentioned in the answer.
```

**Root Cause:**
```python
# main.py:3456 - Answer generated independently
summary = llm_summarize(q.query, preview, stats_hint, lang_instruction)

# main.py:3574-3649 - Chart type selected by rules, not by answer content
# No access to what the LLM wrote in the answer!
chart_type = "line"  # Rule-based decision
if has_time and has_categories:
    if "share" in dims:
        chart_type = "stackedbar"
```

**Evidence:**
- Line 3457: Answer generated first
- Line 3466: Chart data/type selected AFTER answer
- No passing of answer content to chart logic
- Chart doesn't know what variables LLM mentioned

### Problem 2: Chart Type Selection Too Complex

**Location:** main.py:3574-3649

**Issue:**
90 lines of nested if-else for chart type selection:
```python
if has_time and has_categories:
    if "share" in dims:
        chart_type = "stackedbar"
    elif any(d in dims for d in ["price_tariff", "energy_qty", "index", "xrate"]):
        chart_type = "line"
    else:
        chart_type = "line"
elif has_time and not has_categories:
    chart_type = "line"
elif not has_time and has_categories:
    if "share" in dims and len(category_cols) == 1:
        unique_cats = df[category_cols[0]].nunique()
        if unique_cats <= 8:
            chart_type = "pie"
        else:
            chart_type = "bar"
# ... continues for 90 lines
```

**Problems:**
- Hard to maintain
- Doesn't consider what the answer actually discusses
- Might pick "line" when answer talks about composition (should be stacked area)
- Might pick "bar" when answer talks about trend (should be line)

### Problem 3: No Validation Chart Supports Answer

**Missing:**
- Check if variables mentioned in answer are in chart
- Check if chart type matches answer narrative style
- Feedback if critical variables are missing

**Example:**
```
Answer mentions: "price increased due to import share and exchange rate"
Chart shows: Only price over time
Missing: import share, exchange rate

→ User sees chart that doesn't explain the "why"
```

### Recommendations:

#### Fix 1: Generate Chart Guidance in Answer Step (CRITICAL)
```python
def llm_summarize_with_chart_plan(query, data, stats_hint, lang):
    """Generate answer AND specify what should be in the chart."""

    prompt = f"""
    Generate answer to user query and specify chart requirements.

    Output format:
    ANSWER:
    [Your answer text here]

    CHART_VARIABLES:
    [Comma-separated list of variables that should be in chart to support the answer]

    CHART_TYPE_SUGGESTION:
    [line/bar/stackedbar/pie - based on what you discussed in the answer]
    """

    response = llm.invoke(prompt)

    # Parse response
    answer = extract_section(response, "ANSWER")
    chart_vars = extract_section(response, "CHART_VARIABLES").split(",")
    chart_type_suggestion = extract_section(response, "CHART_TYPE_SUGGESTION")

    return answer, chart_vars, chart_type_suggestion
```

#### Fix 2: Validate Chart Includes Key Variables
```python
def validate_chart_supports_answer(answer_text: str, chart_data: pd.DataFrame) -> dict:
    """Check if chart includes variables mentioned in answer."""

    # Extract variable mentions from answer
    mentioned_vars = extract_variable_mentions(answer_text)

    # Check if chart includes them
    chart_columns = set(chart_data.columns)
    missing_vars = [v for v in mentioned_vars if v not in chart_columns]

    return {
        "is_valid": len(missing_vars) == 0,
        "missing_variables": missing_vars,
        "suggestion": f"Add {missing_vars} to chart" if missing_vars else "OK"
    }
```

#### Fix 3: Simplify Chart Type Selection
```python
# Use LLM suggestion + simple validation rules
def select_chart_type(
    suggested_type: str,
    df: pd.DataFrame,
    has_time: bool,
    has_categories: bool
) -> str:
    """Select chart type based on LLM suggestion + data structure validation."""

    # Validate suggestion makes sense for data structure
    if suggested_type == "line" and not has_time:
        return "bar"  # Can't do line without time

    if suggested_type == "pie" and (has_time or df[category_cols[0]].nunique() > 10):
        return "bar"  # Pie doesn't work for time series or many categories

    return suggested_type  # Trust LLM if valid
```

---

## 4. Code Structure Issues 🟡 HIGH

### Problem: Monolithic Architecture

**main.py:** 167 KB, ~3,900 lines

**Responsibilities in one file:**
1. FastAPI route handlers (4 endpoints)
2. LLM chain management (3 functions)
3. SQL generation & validation
4. Database connection & query execution
5. Data processing (filtering, stats, shares)
6. Chart type selection & data prep
7. Caching logic
8. Metrics/observability
9. Language detection
10. Share pivot calculations
11. Seasonal decomposition
12. Entity contribution analysis

**Issues:**
- **Hard to test:** Can't unit test chart logic without loading entire app
- **Hard to review:** 3,900 lines in one file
- **Merge conflicts:** Multiple devs editing same file
- **Import cycles:** Everything in one namespace
- **Performance:** Loading massive file on every request

**Recommended Structure:**
```
langchain_railway/
├── main.py (200 lines - FastAPI app only)
├── config.py (environment, constants)
├── models.py (Pydantic models)
├── core/
│   ├── llm.py (LLM chain functions)
│   ├── sql_generator.py (SQL generation with validation)
│   ├── query_executor.py (database execution)
│   └── cache.py (caching logic)
├── analysis/
│   ├── shares.py (share calculations)
│   ├── seasonal.py (seasonal decomposition)
│   ├── stats.py (quick_stats, trends)
│   └── entity_contributions.py
├── visualization/
│   ├── chart_selector.py (chart type logic)
│   ├── chart_builder.py (data formatting)
│   └── chart_validator.py (validate chart supports answer)
├── utils/
│   ├── language.py (language detection)
│   ├── metrics.py (observability)
│   └── domain_knowledge.py (moved from root)
└── tests/
    ├── test_sql_generation.py
    ├── test_aggregations.py
    ├── test_chart_selection.py
    └── test_share_calculations.py
```

**Benefits:**
- Each module < 500 lines
- Easy to unit test
- Clear separation of concerns
- Parallel development possible
- Faster imports (load only what's needed)

---

## 5. Specific Code Quality Issues

### Issue 5.1: No Validation of Total Calculations

**Location:** SQL execution doesn't validate if totals are correct

**Fix:** Add post-execution validation
```python
def validate_total_calculation(sql: str, result: pd.DataFrame, intent: dict) -> bool:
    """Validate that total calculations are correct."""

    if intent["needs_total"]:
        # Check if result has only 1 row (total) or multiple rows (breakdown)
        if intent["needs_breakdown"]:
            return len(result) > 1  # Should have multiple rows
        else:
            return len(result) == 1  # Should have single total row

    return True
```

### Issue 5.2: Share Calculations Not Documented

**Location:** `build_balancing_correlation_df()` (main.py:677-752)

**Issue:** Complex share calculation logic with NO comments

**Fix:** Add docstring with examples
```python
def build_balancing_correlation_df(conn) -> pd.DataFrame:
    """
    Build monthly decomposition of balancing price with entity shares.

    Calculates for each month:
    - share_import: qty_import / total_balancing_qty
    - share_deregulated_hydro: qty_dereg_hydro / total_balancing_qty
    - ... (7 more shares)

    IMPORTANT: Only uses segment='balancing_electricity' quantities.

    Returns:
        DataFrame with columns:
        - date, p_bal_gel, p_bal_usd, xrate
        - share_import, share_deregulated_hydro, share_regulated_hpp, etc.
        - enguri_tariff_gel, gardabani_tpp_tariff_gel, etc.

    Example:
        df = build_balancing_correlation_df(conn)
        # 2024-01: p_bal_gel=65, share_import=0.25, share_hydro=0.45
    """
```

### Issue 5.3: Chart Selection Logic Unclear

**Location:** main.py:3574-3649 (90 lines of nested ifs)

**Fix:** Extract to decision table
```python
CHART_TYPE_DECISION_TABLE = [
    # (condition_fn, chart_type, description)
    (lambda ctx: ctx["has_time"] and ctx["has_share"], "stackedbar", "Time series of composition"),
    (lambda ctx: ctx["has_time"] and not ctx["has_categories"], "line", "Single time series"),
    (lambda ctx: ctx["has_time"] and ctx["has_categories"], "line", "Multi-line trend"),
    (lambda ctx: not ctx["has_time"] and ctx["has_share"] and ctx["n_cats"] <= 8, "pie", "Composition snapshot"),
    (lambda ctx: not ctx["has_time"] and ctx["has_categories"], "bar", "Categorical comparison"),
]

def select_chart_type(df: pd.DataFrame, dims: set) -> str:
    """Select chart type using decision table."""
    ctx = {
        "has_time": has_time_column(df),
        "has_categories": has_category_column(df),
        "has_share": "share" in dims,
        "n_cats": count_unique_categories(df),
    }

    for condition_fn, chart_type, description in CHART_TYPE_DECISION_TABLE:
        if condition_fn(ctx):
            log.info(f"Chart type: {chart_type} ({description})")
            return chart_type

    return "line"  # Default fallback
```

---

## 6. Testing Gaps

### Current State:
- Manual evaluation system (75 test queries)
- No unit tests for core calculation logic
- No tests for SQL generation accuracy
- No tests for chart-answer consistency

### Required Tests:

#### 6.1: Aggregation Tests
```python
def test_total_generation_sql():
    """Test that 'total generation' query generates correct SUM SQL."""
    query = "What was total generation in 2023?"
    sql = generate_sql(query)

    assert "SUM(" in sql.upper()
    assert "GROUP BY" not in sql.upper()  # Should be single total, not breakdown

def test_generation_by_technology_sql():
    """Test that 'by technology' query generates correct GROUP BY."""
    query = "Show me generation by technology in 2023"
    sql = generate_sql(query)

    assert "GROUP BY" in sql.upper()
    assert "type_tech" in sql
```

#### 6.2: Share Calculation Tests
```python
def test_share_calculation_adds_to_100():
    """Test that all shares add up to 100%."""
    df = build_balancing_correlation_df(test_conn)

    share_cols = [c for c in df.columns if c.startswith("share_")]
    total_share = df[share_cols].sum(axis=1)

    assert all(0.99 <= s <= 1.01 for s in total_share), "Shares should add to 100%"
```

#### 6.3: Chart-Answer Consistency Tests
```python
def test_chart_includes_mentioned_variables():
    """Test that chart includes variables mentioned in answer."""
    query = "Why did balancing price increase?"
    answer, chart_data, chart_type = process_query(query)

    # If answer mentions "import share", chart should include it
    if "import share" in answer.lower():
        assert any("import" in c.lower() for c in chart_data.columns)
```

---

## 7. Priority Action Items

### CRITICAL (Fix Immediately):
1. ✅ **Add aggregation intent detection** - Detect if user wants total/average/breakdown
2. ✅ **Validate SQL matches intent** - Ensure SUM is used for totals
3. ✅ **Add aggregation examples to few-shot** - Show LLM how to do totals correctly
4. ✅ **Link chart variables to answer** - LLM specifies what should be in chart
5. ✅ **Validate chart supports answer** - Check mentioned variables are included

### HIGH (Fix This Sprint):
6. ✅ **Consolidate MD files** - Reduce from 7 to 3 files in docs/
7. ✅ **Refactor main.py** - Split into modules (core/, analysis/, visualization/)
8. ⬜ **Add unit tests** - Test aggregations, shares, chart selection
9. ⬜ **Document share calculations** - Add docstrings with examples

### MEDIUM (Fix Next Sprint):
10. ⬜ **Create materialized view for shares** - Move Python share logic to database
11. ⬜ **Add SQL validation layer** - Validate GROUP BY, SUM, AVG logic
12. ⬜ **Simplify chart selection** - Use decision table instead of nested ifs
13. ⬜ **Add integration tests** - Test full query → answer → chart flow

---

## 8. Metrics to Track Improvement

### Before Fixes:
- **SQL Accuracy:** Unknown (no validation)
- **Chart-Answer Match:** ~60% (estimated, based on manual review)
- **Code Maintainability:** main.py = 3,900 lines (unmaintainable)
- **Doc Redundancy:** 7 files, ~40% overlap

### Target After Fixes:
- **SQL Accuracy:** 95%+ (with validation)
- **Chart-Answer Match:** 90%+ (with LLM chart guidance)
- **Code Maintainability:** Largest module < 500 lines
- **Doc Redundancy:** 3 files, < 10% overlap

---

## 9. Estimated Effort

| Task | Effort | Impact |
|------|--------|--------|
| Add aggregation intent detection | 2 hours | CRITICAL |
| Validate SQL matches intent | 3 hours | CRITICAL |
| Link chart to answer (LLM guidance) | 4 hours | CRITICAL |
| Consolidate MD files | 2 hours | HIGH |
| Refactor main.py into modules | 8 hours | HIGH |
| Add unit tests for aggregations | 4 hours | HIGH |
| Create share materialized view | 3 hours | MEDIUM |
| Simplify chart selection logic | 3 hours | MEDIUM |

**Total:** ~29 hours (~4 days)

---

## 10. Conclusion

The energy chatbot has **solid domain knowledge integration** and **comprehensive evaluation coverage**, but suffers from:

1. **🔴 CRITICAL: Unreliable SQL generation for totals/aggregations**
   - Fix: Add intent detection + SQL validation

2. **🔴 CRITICAL: Chart-answer mismatch**
   - Fix: LLM specifies chart variables in answer step

3. **🟡 HIGH: Monolithic architecture**
   - Fix: Refactor into modules (core/, analysis/, visualization/)

4. **🟡 HIGH: Documentation overload**
   - Fix: Consolidate 7 MD files → 3 organized files

**Recommendation:** Prioritize fixes 1-5 (CRITICAL) in this sprint, then tackle refactoring (HIGH) in next sprint.
