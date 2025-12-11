# Testing Guide - Manual LLM Quality Evaluation

## Overview

This guide explains how to use the **manual test suite** to evaluate LLM response quality for the Georgian electricity market chatbot.

**Purpose:** Systematically test responses to ensure:
- ✅ SQL uses correct tables and columns
- ✅ Answers are accurate and well-formatted
- ✅ Critical rules are followed (seasonality, thermal imports, column names)
- ✅ Charts are appropriate and clear

**Test Suite:** `evaluation/test_suite.py` contains 30+ test cases across 10 categories

---

## Quick Start

### 1. List All Test Cases
```bash
python evaluation/test_suite.py
```

Output:
```
Total test cases: 30+
Available categories:
  - energy_security (3 tests)
  - balancing_price (4 tests)
  - demand (3 tests)
  - tariffs (3 tests)
  - support_schemes (2 tests)
  ...
```

### 2. Run Single Test Case
```bash
python evaluation/test_suite.py --test 1
```

This will print:
- Query to test
- Expected results (tables, columns, should_mention keywords)
- Critical rules to check
- Curl command to run the query
- Scoring rubric

### 3. Run Category Tests
```bash
python evaluation/test_suite.py --category energy_security
```

Runs all energy security test cases sequentially.

### 4. Run All Tests
```bash
python evaluation/test_suite.py --run-all
```

**Warning:** This will run 30+ tests. Use for comprehensive regression testing.

---

## Test Case Structure

Each test case includes:

```python
TestCase(
    id=1,
    category=Category.ENERGY_SECURITY,
    query="როგორ შეაფასებ საქართველოს ენერგეტიკულ უსაფრთხოებას?",
    description="Energy security assessment in Georgian",
    expected=ExpectedResult(
        expected_tables=["trade_by_source"],
        language="ka",
        should_mention=["იმპორტი", "ჰიდრო", "თერმული", "გაზი"],
        should_clarify_thermal_imports=True,  # CRITICAL
        should_have_seasonal_breakdown=True
    ),
    notes="Must clarify that thermal uses imported gas"
)
```

---

## Manual Testing Process

### Step 1: Select Test Case
```bash
python evaluation/test_suite.py --test 1
```

### Step 2: Run Query
Copy the curl command from output:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "როგორ შეაფასებ საქართველოს ენერგეტიკულ უსაფრთხოებას?"}'
```

Or use your frontend/Postman.

### Step 3: Review Response

Check the JSON response for:
1. **Plan** - Does intent match expected?
2. **SQL** - Are tables/columns correct?
3. **Answer** - Is it accurate and well-formatted?
4. **Chart** - Is it appropriate?

### Step 4: Score Each Aspect

Use the scoring rubric (1-5 for each):

#### 1. SQL CORRECTNESS (1-5)
- ✅ Uses expected tables? (e.g., `tech_quantity_view` for demand)
- ✅ Has required columns? (e.g., `time_month`, `quantity`)
- ✅ Avoids forbidden tables? (e.g., don't use `trade_derived_entities` for simple demand)
- ✅ Has correct filters? (e.g., demand type_tech values)
- ✅ Proper aggregation? (e.g., `GROUP BY time_month` for trends)

**Score:**
- 5 = Perfect
- 4 = Minor issues (unnecessary column)
- 3 = Works but suboptimal (wrong table but gets result)
- 2 = Major issues (wrong aggregation)
- 1 = Completely wrong or fails

#### 2. ANSWER QUALITY (1-5)
- ✅ Correct information?
- ✅ Clear and concise?
- ✅ Well-structured (opening, evidence, explanation)?
- ✅ Includes relevant drivers/factors?

**Score:**
- 5 = Excellent
- 4 = Good, minor issues
- 3 = Acceptable
- 2 = Poor (inaccurate or missing info)
- 1 = Wrong information

#### 3. LANGUAGE MATCH (1-5)
- ✅ Correct language? (Georgian query → Georgian answer)
- ✅ Natural phrasing?
- ✅ No mixed languages?

#### 4. FORMAT COMPLIANCE (1-5)
- ✅ Numbers have thousand separators? (1,234 not 1234)
- ✅ Percentages have one decimal? (15.3% not 15%)
- ✅ Units included? (GEL/MWh, thousand MWh, %)
- ✅ Trends include direction + magnitude + timeframe?
- ✅ No raw column names? (use "balancing price" not "p_bal_gel")

### Step 5: Check Critical Rules (Pass/Fail)

#### Rule 1: Energy Security
**For queries about energy security:**
- ❓ Does answer clarify thermal uses imported gas?
- ❓ Does it avoid saying "thermal is local" or "thermal reduces import dependence"?

**Pass/Fail:** _______

#### Rule 2: Seasonal Breakdown
**For price comparison queries:**
- ❓ Does answer show summer AND winter averages separately?
- ❓ Does it avoid using annual average only?

**Pass/Fail:** _______

#### Rule 3: Column Names
**For all answers:**
- ❓ Does answer use descriptive names (not raw database columns)?
- ❓ No "p_bal_gel" or "share_import" in user-facing text?

**Pass/Fail:** _______

#### Rule 4: Table Selection
**For demand/generation queries:**
- ❓ Uses `tech_quantity_view` (not `trade_derived_entities`)?

**Pass/Fail:** _______

### Step 6: Calculate Overall Score
```
Overall Score = (SQL Score + Answer Score + Language Score + Format Score) / 4

Adjust down by 1 point for each critical rule failure.
```

### Step 7: Document Results

Create a test log file:
```
evaluation/results/2024-12-11_test_run.md
```

Format:
```markdown
# Test Run: 2024-12-11

## Test Case #1: Energy Security Assessment (Georgian)

**Query:** როგორ შეაფასებ საქართველოს ენერგეტიკულ უსაფრთხოებას?

**Scores:**
- SQL Correctness: 5/5
- Answer Quality: 4/5 (missing gas import emphasis)
- Language Match: 5/5
- Format Compliance: 4/5 (no thousand separators)

**Critical Rules:**
- Thermal import clarification: ❌ FAIL (didn't mention gas)
- Seasonal breakdown: ✅ PASS
- Column names: ✅ PASS

**Overall: 3.5/5** (Adjusted down due to critical rule failure)

**Issues Found:**
1. Answer doesn't clarify thermal uses imported gas
2. Numbers lack thousand separators (171000 instead of 171,000)

**Action Items:**
- Check if EnergySecurityAnalysis domain knowledge is loading
- Verify guidance section triggers for energy security queries
```

---

## Test Categories

### 1. Energy Security (3 tests)
**Focus:** Import dependence, thermal gas imports, seasonal vulnerability

**Critical Rule:** Must clarify thermal uses imported gas

**Examples:**
- Test #1: "როგორ შეაფასებ ენერგეტიკულ უსაფრთხოებას?"
- Test #2: "What is Georgia's import dependence in winter?"
- Test #3: "Can Georgia achieve energy independence?"

### 2. Balancing Price (4 tests)
**Focus:** Price drivers, trends, summer/winter comparison

**Critical Rules:**
- Must show summer AND winter prices separately
- Should explain composition changes first

**Examples:**
- Test #11: "რატომ გაიზარდა საბალანსო ფასი 2024 წელს?"
- Test #12: "Compare summer and winter balancing prices 2023"
- Test #13: "What drives balancing electricity price?"

### 3. Demand (3 tests)
**Focus:** Demand trends, growth rates, seasonal patterns

**Critical Rule:** Must use `tech_quantity_view` (not `trade_derived_entities`)

**Examples:**
- Test #21: "Show me total demand from 2020 to 2023"
- Test #22: "როგორია მოთხოვნის ზრდის ტემპი?"
- Test #23: "What causes demand fluctuations?"

### 4. Tariffs (3 tests)
**Focus:** Hydro vs thermal comparison, regulated entities, trends

**Examples:**
- Test #31: "შეადარე ჰიდრო და თერმული ტარიფები"
- Test #32: "How many regulated entities are there?"
- Test #33: "Show Enguri tariff trends 2020-2024"

### 5. Support Schemes (2 tests)
**Focus:** CfD/PPA explanation, renewable PPA trends

**Critical Rule:** Conceptual questions should skip SQL (Option 4)

**Examples:**
- Test #41: "რა არის CfD სქემა?" (should skip SQL)
- Test #42: "Show renewable PPA share trends"

---

## Tracking Improvements Over Time

### Create Baseline (First Run)
```bash
python evaluation/test_suite.py --run-all > evaluation/results/baseline_2024-12-11.txt
```

Manually score all tests and document:
```
Average SQL Correctness: 4.2/5
Average Answer Quality: 3.8/5
Critical Rule Pass Rate: 65% (13/20 rules passed)

Top Issues:
1. Thermal import clarification: 33% pass rate
2. Seasonal breakdown: 75% pass rate
3. Column names: 90% pass rate
```

### After Making Changes (e.g., adding few-shot examples)
```bash
python evaluation/test_suite.py --run-all > evaluation/results/after_examples_2024-12-12.txt
```

Compare:
```
BEFORE (2024-12-11):
  SQL Correctness: 4.2/5
  Answer Quality: 3.8/5
  Critical Rules: 65%

AFTER (2024-12-12):
  SQL Correctness: 4.6/5 (+0.4) ✅
  Answer Quality: 4.3/5 (+0.5) ✅
  Critical Rules: 85% (+20%) ✅

Improvement: YES, few-shot examples helped significantly
```

### Track Patterns
Create a spreadsheet:
```
Date       | Avg SQL | Avg Answer | Critical Rules | Notes
-----------|---------|------------|----------------|------------------
2024-12-11 | 4.2     | 3.8        | 65%           | Baseline
2024-12-12 | 4.6     | 4.3        | 85%           | Added examples
2024-12-15 | 4.7     | 4.5        | 90%           | Added format rules
```

---

## Advanced: Automated Testing (Future)

Once you're comfortable with manual testing, you can add automation:

```python
# evaluation/automated_tests.py

def validate_sql(response, expected):
    """Automatically check SQL correctness"""
    sql = response["sql"].lower()

    # Check tables
    for table in expected.expected_tables:
        if table not in sql:
            return False, f"Missing expected table: {table}"

    # Check forbidden tables
    if expected.forbidden_tables:
        for table in expected.forbidden_tables:
            if table in sql:
                return False, f"Used forbidden table: {table}"

    return True, "SQL looks good"


def validate_answer(response, expected):
    """Automatically check answer quality"""
    answer = response["answer"].lower()

    # Check language
    if expected.language == "ka":
        # Georgian characters: ა-ჰ
        if not re.search(r'[ა-ჰ]', answer):
            return False, "Answer should be in Georgian"

    # Check should_mention keywords
    if expected.should_mention:
        for keyword in expected.should_mention:
            if keyword.lower() not in answer:
                return False, f"Missing expected keyword: {keyword}"

    # Check should_not_mention
    if expected.should_not_mention:
        for keyword in expected.should_not_mention:
            if keyword.lower() in answer:
                return False, f"Contains forbidden phrase: {keyword}"

    return True, "Answer looks good"


# Run automated checks
for test_case in ALL_TEST_CASES:
    response = run_query(test_case.query)
    sql_ok, sql_msg = validate_sql(response, test_case.expected)
    answer_ok, answer_msg = validate_answer(response, test_case.expected)

    if not sql_ok or not answer_ok:
        print(f"FAIL: Test #{test_case.id}")
        print(f"  SQL: {sql_msg}")
        print(f"  Answer: {answer_msg}")
```

---

## Tips for Effective Testing

### 1. Start Small
- Don't run all 30+ tests at once initially
- Start with 1 category (e.g., energy_security)
- Get familiar with the process

### 2. Focus on Critical Rules
- The scoring is useful, but critical rules are MORE important
- A response with score 4/5 that fails critical rules is WORSE than 3.5/5 that passes them

### 3. Look for Patterns
- If multiple tests in same category fail similarly, there's a systematic issue
- Example: All energy security tests fail thermal import check → domain knowledge not loading

### 4. Test After Each Change
- Added few-shot examples? Run tests
- Changed domain knowledge? Run tests
- Modified prompts? Run tests

### 5. Document Everything
- Save test results with dates
- Note what changed between runs
- Track improvement trends

### 6. Prioritize by User Impact
- Energy security tests (#1-3): HIGH priority (common user concern)
- Balancing price tests (#11-14): HIGH priority (core use case)
- Ownership tests: MEDIUM priority (less frequent)

---

## Common Issues & Solutions

### Issue 1: Wrong Table Selection
**Symptom:** Uses `trade_derived_entities` for simple demand query

**Solution:**
- Check `TableSelectionGuidance` in domain_knowledge.py
- Add few-shot examples showing correct table usage
- Verify trigger keywords in get_relevant_domain_knowledge()

### Issue 2: Missing Seasonal Breakdown
**Symptom:** Shows annual average instead of summer/winter

**Solution:**
- Check `PriceComparisonRules` loading
- Add trigger keyword "price trend" → PriceComparisonRules
- Add explicit guidance in system prompt

### Issue 3: Doesn't Clarify Thermal Imports
**Symptom:** Treats thermal as local generation

**Solution:**
- Check `EnergySecurityAnalysis` loading
- Verify triggers: "უსაფრთხოება", "energy security", "independence"
- Add energy security guidance section (already done in recent commit)

### Issue 4: Uses Raw Column Names
**Symptom:** Answer contains "p_bal_gel" or "share_import"

**Solution:**
- Check column name enforcement in focus rules
- Add examples in few-shot prompts with correct terminology
- Add validation in post-processing (future)

---

## Next Steps

1. **Run Baseline Tests** (Today)
   ```bash
   python evaluation/test_suite.py --category energy_security
   ```

2. **Implement Few-Shot Examples** (Next)
   - Add examples to llm_generate_plan_and_sql()
   - See TUNING_REVIEW.md Phase 1

3. **Re-run Tests & Compare** (After implementing)
   ```bash
   python evaluation/test_suite.py --category energy_security
   ```

4. **Expand Test Coverage** (Ongoing)
   - Add more test cases as you find edge cases
   - Add tests for new features

5. **Automate** (Future)
   - Start with simple automated checks
   - Gradually add more sophisticated validation
   - Eventually: CI/CD integration

---

## Questions?

If you find issues with test cases or have suggestions:
1. Document in test_suite.py notes field
2. Create new test cases for edge cases you discover
3. Track in evaluation/results/ directory

Happy testing! 🎯
