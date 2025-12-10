# Evaluation System Guide

**Purpose**: Systematically test query generation and answer quality to validate that Phase 1 optimizations (caching, conditional guidance, merged LLM calls) don't degrade quality.

---

## Files

### 1. `evaluation_dataset.json`
Comprehensive test dataset with 75 queries covering:
- **Single value queries** (23) - Simple lookups: "What was balancing price in June 2024?"
- **List queries** (11) - Entity/type listings: "List all entities selling on balancing market"
- **Comparison queries** (15) - Entity/period comparisons: "Compare tariffs for regulated vs deregulated"
- **Trend queries** (19) - Time series analysis: "Balancing price trend over 2023-2024"
- **Analyst queries** (7) - Deep analysis: "What drives balancing price changes?"

Each query includes:
- Natural language query (English, Georgian, Russian)
- Expected SQL patterns (tables, joins, aggregations)
- Quality criteria (must include/exclude, sentence count, language)
- Performance expectations

### 2. `test_evaluation.py`
Automated test runner that:
- Executes queries against the API
- Validates SQL generation correctness
- Validates answer quality against criteria
- Measures performance
- Generates detailed reports

### 3. `EVALUATION_GUIDE.md` (this file)
Documentation for using the evaluation system

---

## Usage

### Quick Test (10 representative queries)
```bash
python test_evaluation.py --mode quick
```
**Expected time**: 1-2 minutes
**Purpose**: Quick smoke test after changes

### Full Test (all 75 queries)
```bash
python test_evaluation.py --mode full
```
**Expected time**: 10-15 minutes
**Purpose**: Comprehensive validation before deployment

### Test Specific Query Type
```bash
# Test only single value queries (simple lookups)
python test_evaluation.py --type single_value

# Test only analytical queries (most complex)
python test_evaluation.py --type analyst

# Test only trend queries
python test_evaluation.py --type trend
```

### Test Specific Query
```bash
python test_evaluation.py --query sv_001
```

### Save Results to File
```bash
python test_evaluation.py --mode full --output results_2025-11-11.json
```

---

## Configuration

Set environment variables or use `.env` file:
```bash
API_URL=http://localhost:8000/ask
APP_SECRET_KEY=your_secret_key
```

---

## Interpreting Results

### Overall Metrics
```
Overall Results:
  Total queries: 75
  Passed: 68 (90.7%)
  Failed: 7 (9.3%)
```

**Target pass rate**: >90% for production deployment

### Results by Query Type
```
Results by Query Type:
  single_value   : 22/23 passed ( 95.7%)
  list           : 11/11 passed (100.0%)
  comparison     : 14/15 passed ( 93.3%)
  trend          : 17/19 passed ( 89.5%)
  analyst        :  4/ 7 passed ( 57.1%)
```

**Expected patterns**:
- Single value / List: 95-100% pass rate (simple queries, well-optimized)
- Comparison / Trend: 85-95% pass rate (moderate complexity)
- Analyst: 70-90% pass rate (complex, needs full guidance)

**Warning signs**:
- Single value <90% → Basic functionality broken
- Analyst <60% → Guidance reduction went too far

### Performance Metrics
```
Performance:
  Average response time: 8500ms
  Simple queries avg:    3200ms (target: <8s)
  Complex queries avg:   18500ms (target: <45s)
```

**Interpretation**:
- Simple queries: <5s = excellent, 5-8s = good, >8s = investigate
- Complex queries: <20s = excellent, 20-35s = good, >45s = too slow

### Issue Breakdown
```
Issue Breakdown:
  SQL pattern issues:    3
  Quality issues:        4
  Performance issues:    2
```

**SQL pattern issues**: LLM generated SQL missing expected tables/joins → May need prompt improvement
**Quality issues**: Answer missing required information → Guidance may be insufficient
**Performance issues**: Exceeds time thresholds → Optimization needed or query too complex

---

## Quality Criteria Explained

### Must Include
Items that MUST appear in the answer:
- **Numeric values**: "45.2 GEL/MWh" not just "increased"
- **Units**: "GEL/MWh" or "MWh" not just raw numbers
- **Time periods**: "June 2024" not just "recently"
- **Context**: For analyst queries, must mention drivers (composition, xrate, seasonal)

### Must NOT Include
Items that should NOT appear:
- **Confidential info**: PPA pricing details
- **Irrelevant details**: For simple queries, no detailed analysis
- **Hallucinations**: Made-up entities or dates

### Sentence Count
- **Single value / List**: 1-3 sentences (concise)
- **Comparison**: 3-5 sentences (brief comparison)
- **Trend**: 5-8 sentences (pattern description)
- **Analyst**: 8-12 sentences (comprehensive analysis)

### Language Matching
If query in Georgian, answer must be in Georgian
If query in Russian, answer must be in Russian
If query in English, answer in English

---

## Adding New Test Queries

Edit `evaluation_dataset.json` and add:

```json
{
  "id": "your_id_001",
  "type": "single_value",
  "language": "en",
  "query": "Your natural language question",
  "expected_sql_patterns": [
    "table_name",
    "WHERE condition",
    "aggregate_function"
  ],
  "expected_tables": ["table1", "table2"],
  "quality_criteria": {
    "must_include": ["item1", "item2"],
    "must_not_include": ["unwanted_item"],
    "max_sentences": 2
  },
  "expected_performance": {
    "total_time_ms": "< 8000"
  }
}
```

---

## CI/CD Integration

### Pre-deployment Test
```bash
# Run full test suite and fail if pass rate < 90%
python test_evaluation.py --mode full --output results.json

# Parse results and exit with error if failed
pass_rate=$(jq '.results | map(select(.status == "pass")) | length' results.json)
total=$(jq '.results | length' results.json)
if [ $(echo "$pass_rate / $total < 0.9" | bc -l) -eq 1 ]; then
  echo "FAIL: Pass rate below 90%"
  exit 1
fi
```

### Regression Testing
Run before/after optimization to detect quality degradation:
```bash
# Before optimization
python test_evaluation.py --mode full --output baseline.json

# After optimization
python test_evaluation.py --mode full --output optimized.json

# Compare results
python compare_results.py baseline.json optimized.json
```

---

## Expected Results After Phase 1 Optimizations

### Phase 1: Response Caching + Conditional Guidance
**Expected improvements**:
- Simple queries: 26s → 3-5s (5-8x faster) ✓
- Complex queries: 26s → 26s (same, but cached repeats <0.5s) ✓
- Pass rate: Should maintain >90%

**Risk areas**:
- Analyst queries might fail if guidance reduced too much
- Quality validation: Check "must_include" criteria for driver analysis

### Phase 1B: Cache All LLM Calls
**Expected improvements**:
- Repeated queries: <0.3s (98% faster) ✓
- First-time queries: Same as Phase 1
- Pass rate: Should maintain >90%

### Phase 1C: Merge Domain Reasoning → SQL
**Expected improvements**:
- All queries: ~6s faster (12% improvement) ✓
- Simple queries: 5s → 4s
- Complex queries: 26s → 20s
- Pass rate: Should maintain >90% (merged reasoning might slightly affect quality)

**Risk areas**:
- SQL generation quality might degrade if reasoning not properly merged
- Watch for missing JOIN or WHERE clauses

### Phase 1D: Security Hardening
**Expected impact**:
- Performance: No impact (read-only is enforcement only)
- Rate limiting: May affect burst testing (add delays between requests)
- Query timeout: Might fail on very slow queries (>30s)
- Pass rate: Should maintain >90%

---

## Troubleshooting

### All Tests Failing with Connection Error
**Issue**: Can't connect to API
**Fix**: Check `API_URL` in `.env` and verify server is running

### High SQL Pattern Failure Rate
**Issue**: LLM not generating expected SQL
**Possible causes**:
1. Guidance reduction removed critical examples
2. Schema knowledge not loaded properly
3. Table whitelisting too restrictive

**Fix**: Review llm_generate_plan_and_sql() prompt

### High Quality Failure Rate
**Issue**: Answers missing required information
**Possible causes**:
1. Guidance reduction removed domain knowledge
2. max_tokens limit cutting off answers
3. Conditional guidance logic too aggressive

**Fix**: Review needs_full_guidance() logic and token limits

### Performance Issues
**Issue**: All queries >45s
**Possible causes**:
1. Railway network issues
2. LLM API throttling
3. Database query slow

**Fix**: Check /metrics endpoint for bottleneck

---

## Metrics to Track Over Time

Create a metrics dashboard tracking:
1. **Pass rate by query type** (trend over versions)
2. **Average response time by type** (detect performance regressions)
3. **Cache hit rate** (should be >60% in production)
4. **SQL pattern accuracy** (should be >95%)
5. **Quality score** (should be >90%)

---

## Next Steps

After validation:
1. ✅ Run baseline test before Phase 1 (establish quality baseline)
2. ✅ Run after Phase 1 (validate no regression)
3. ✅ Run after Phase 1B (validate caching works)
4. ✅ Run after Phase 1C (validate merged reasoning maintains quality)
5. ✅ Run after Phase 1D (validate security doesn't break functionality)
6. ⏱️ Set up CI/CD to run on every commit
7. ⏱️ Deploy to staging and run full test
8. ⏱️ Deploy to production with monitoring

---

## Contact

For questions about the evaluation system or to report issues:
- Review TIMEOUT_QUALITY_EVALUATION.md for quality impact analysis
- Review BEST_PRACTICES_REVIEW.md for implementation standards
- Check main.py for Phase 1/1B/1C/1D implementation details
