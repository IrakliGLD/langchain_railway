# Energy Chatbot - Evaluation Guide

Complete guide for testing and validating the energy chatbot quality.

---

## Quick Start

### Run Evaluation on Railway (Production)

**Prerequisites:**
1. Railway URL: `https://your-app.railway.app`
2. API Key: Get from Railway Dashboard → Variables → `APP_SECRET_KEY`

**Browser Method (Easiest):**
1. Install [ModHeader](https://chrome.google.com/webstore/detail/modheader/idgpnmonknjnojddfkpgkljpfnnfcklj) extension
2. Add header: `X-App-Key: your_api_key`
3. Visit: `https://your-app.railway.app/evaluate?mode=quick`
4. Check pass rate ≥90%

**cURL Method:**
```bash
export RAILWAY_URL="https://your-app.railway.app"
export API_KEY="your_api_key"

# Quick test (10 queries, 1-2 minutes)
curl -H "X-App-Key: $API_KEY" "$RAILWAY_URL/evaluate?mode=quick"

# Full test (75 queries, 10-15 minutes)
curl -H "X-App-Key: $API_KEY" "$RAILWAY_URL/evaluate?mode=full&format=json" > results.json
```

### Run Evaluation Locally

```bash
# Quick test
python test_evaluation.py --mode quick

# Full test
python test_evaluation.py --mode full

# Test specific type
python test_evaluation.py --type analyst
python test_evaluation.py --type single_value

# Test specific query
python test_evaluation.py --query sv_001
```

---

## Evaluation Dataset

**File:** `evaluation_dataset.json`

**75 test queries** covering:

| Type | Count | Description | Target Pass Rate |
|------|-------|-------------|------------------|
| **single_value** | 23 | Simple lookups: "What was price in June 2024?" | 95-100% |
| **list** | 11 | Entity listings: "List all HPPs" | 95-100% |
| **comparison** | 15 | Comparisons: "Compare regulated vs deregulated tariffs" | 85-95% |
| **trend** | 19 | Time series: "Balancing price trend 2023-2024" | 85-95% |
| **analyst** | 7 | Deep analysis: "What drives balancing price?" | 70-90% |

**Each query includes:**
- Natural language query (English, Georgian, Russian)
- Expected SQL patterns (tables, joins, aggregations)
- Quality criteria (must include/exclude, sentence count)
- Performance expectations

---

## Interpreting Results

### Overall Pass Rate

```
Pass Rate: 92.5%
```

**Status:**
- **≥90%** ✅ Production ready
- **70-89%** ⚠️ Review failures before deploying
- **<70%** ❌ Critical issues - do not deploy

### Results by Query Type

```
single_value   : 22/23 passed (95.7%)
list           : 11/11 passed (100.0%)
comparison     : 14/15 passed (93.3%)
trend          : 17/19 passed (89.5%)
analyst        :  4/ 7 passed (57.1%)
```

**Warning Signs:**
- Single value <90% → Basic functionality broken
- Analyst <60% → Domain guidance insufficient
- All types declining → Recent change broke core functionality

### Performance Metrics

```
Average response time: 8500ms
Simple queries avg:    3200ms (target: <8s)
Complex queries avg:   18500ms (target: <45s)
```

**Thresholds:**
- Simple queries: <5s excellent, 5-8s good, >8s investigate
- Complex queries: <20s excellent, 20-35s good, >45s too slow

### Issue Breakdown

```
Issue Breakdown:
  SQL pattern issues:    3  ← LLM didn't generate expected SQL
  Quality issues:        4  ← Answer missing required info
  Performance issues:    2  ← Exceeded time thresholds
```

---

## Quality Criteria

### Must Include
Items that MUST appear in answers:
- **Numeric values:** "45.2 GEL/MWh" not just "increased"
- **Units:** "GEL/MWh" or "MWh" not raw numbers
- **Time periods:** "June 2024" not "recently"
- **Context:** For analyst queries, mention drivers (composition, xrate, seasonal)

### Must NOT Include
- **Confidential info:** PPA pricing estimates
- **Technical jargon:** SQL terms, table names, schema details
- **Hallucinations:** Made-up entities or dates

### Sentence Count
- **Single value / List:** 1-3 sentences (concise)
- **Comparison:** 3-5 sentences (brief comparison)
- **Trend:** 5-8 sentences (pattern description)
- **Analyst:** 8-12 sentences (comprehensive analysis)

### Language Matching
Response language MUST match query language:
- Georgian query → Georgian answer
- Russian query → Russian answer
- English query → English answer

---

## Test Modes

### Quick Mode (10 queries, 1-2 minutes)
```bash
python test_evaluation.py --mode quick
```
**Use:** After every deployment, quick smoke test

**Sample queries:**
- 2 single_value (basic lookups)
- 2 list (entity enumeration)
- 2 comparison (entity/period comparison)
- 2 trend (time series)
- 2 analyst (deep analysis)

### Full Mode (75 queries, 10-15 minutes)
```bash
python test_evaluation.py --mode full
```
**Use:** Before production releases, comprehensive validation

**Coverage:**
- All query types
- All languages (English, Georgian, Russian)
- All data domains (price, tariff, generation, CPI)

### Type-Specific Tests
```bash
# Test only analyst queries (most complex)
python test_evaluation.py --type analyst

# Test only simple lookups
python test_evaluation.py --type single_value
```
**Use:** When specific feature has issues

---

## Common Commands

### Railway Production

```bash
# Quick smoke test after deployment
curl -H "X-App-Key: $API_KEY" "$RAILWAY_URL/evaluate?mode=quick"

# Full validation before release
curl -H "X-App-Key: $API_KEY" "$RAILWAY_URL/evaluate?mode=full&format=json" > results.json

# Test specific feature
curl -H "X-App-Key: $API_KEY" "$RAILWAY_URL/evaluate?type=analyst"

# Get pass rate for monitoring
curl -H "X-App-Key: $API_KEY" "$RAILWAY_URL/evaluate?mode=quick&format=json" | jq '.summary.pass_rate'
```

### Local Testing

```bash
# Quick test with output file
python test_evaluation.py --mode quick --output results.json

# Full test for specific type
python test_evaluation.py --type trend --output trend_results.json

# Test specific query
python test_evaluation.py --query sv_001
```

---

## Adding New Test Queries

Edit `evaluation_dataset.json`:

```json
{
  "id": "your_id_001",
  "type": "single_value",
  "language": "en",
  "query": "What was balancing price in June 2024?",
  "expected_sql_patterns": [
    "price_with_usd",
    "WHERE",
    "p_bal_gel"
  ],
  "expected_tables": ["price_with_usd"],
  "quality_criteria": {
    "must_include": ["GEL", "June", "2024"],
    "must_not_include": ["SQL", "schema"],
    "max_sentences": 2
  },
  "expected_performance": {
    "total_time_ms": "< 8000"
  }
}
```

**Guidelines:**
1. **ID:** Use format `{type}_{number}` (e.g., `sv_001`, `analyst_007`)
2. **Type:** One of: single_value, list, comparison, trend, analyst
3. **SQL Patterns:** Keywords/tables that SHOULD appear in generated SQL
4. **Quality Criteria:** What MUST/MUST NOT be in answer
5. **Performance:** Expected max response time

---

## Troubleshooting

### All Tests Failing with Connection Error
**Issue:** Can't connect to API
**Fix:**
1. Check `API_URL` in `.env`
2. Verify server is running: `curl http://localhost:8000/ask`
3. Check API key is correct

### High SQL Pattern Failure Rate
**Issue:** LLM not generating expected SQL
**Possible Causes:**
- Guidance reduction removed critical examples
- Schema knowledge not loaded properly
- Table whitelisting too restrictive

**Fix:**
1. Review `llm_generate_plan_and_sql()` prompt in main.py
2. Check few-shot SQL examples include similar queries
3. Verify table whitelist in `validate_and_fix_sql()`

### High Quality Failure Rate
**Issue:** Answers missing required information
**Possible Causes:**
- Domain guidance reduction too aggressive
- max_tokens limit cutting off answers
- Conditional guidance logic incorrect

**Fix:**
1. Review `needs_full_guidance()` logic
2. Check domain knowledge is loaded
3. Increase max_tokens if answers are truncated

### Performance Issues
**Issue:** All queries >45s
**Possible Causes:**
- Railway network latency
- LLM API throttling
- Database query slow
- No caching

**Fix:**
1. Check `/metrics` endpoint for bottleneck
2. Review Railway logs
3. Verify LLM cache is working
4. Check database query performance

### "401 Unauthorized" on Railway
**Issue:** API key is wrong or missing
**Fix:**
1. Check `X-App-Key` header matches Railway `APP_SECRET_KEY`
2. Verify header is correctly configured in ModHeader
3. Test with cURL to isolate browser vs key issue

---

## CI/CD Integration

### Pre-deployment Test
```bash
#!/bin/bash
# Run full test suite and fail if pass rate < 90%

python test_evaluation.py --mode full --output results.json

pass_rate=$(jq '.results | map(select(.status == "pass")) | length' results.json)
total=$(jq '.results | length' results.json)
pass_percentage=$(echo "scale=2; $pass_rate / $total * 100" | bc)

echo "Pass rate: $pass_percentage%"

if (( $(echo "$pass_percentage < 90" | bc -l) )); then
  echo "FAIL: Pass rate below 90%"
  jq '.results[] | select(.status == "fail") | {id, reason}' results.json
  exit 1
fi

echo "PASS: Quality validation successful"
```

### Regression Testing
```bash
# Before optimization
python test_evaluation.py --mode full --output baseline.json

# After optimization
python test_evaluation.py --mode full --output optimized.json

# Compare pass rates
baseline_rate=$(jq '.summary.pass_rate' baseline.json)
optimized_rate=$(jq '.summary.pass_rate' optimized.json)

echo "Baseline: $baseline_rate"
echo "Optimized: $optimized_rate"

# Fail if regression > 5%
if (( $(echo "$optimized_rate < $baseline_rate - 0.05" | bc -l) )); then
  echo "REGRESSION: Pass rate decreased by >5%"
  exit 1
fi
```

---

## Monitoring Checklist

### After Each Deployment
- [ ] Run `mode=quick` (1-2 min)
- [ ] Check pass rate ≥90%
- [ ] Check performance within targets
- [ ] Review any new failures

### Weekly
- [ ] Run `mode=full` (10-15 min)
- [ ] Compare pass rate trend
- [ ] Archive results JSON
- [ ] Review slow queries

### Before Major Release
- [ ] Run full test baseline
- [ ] Make changes
- [ ] Run full test again
- [ ] Compare results
- [ ] No regression >5%

---

## Expected Results After Optimizations

### Phase 1: Caching + Conditional Guidance
- Simple queries: 26s → 3-5s (5-8x faster)
- Complex queries: 26s → 26s (same, but cached repeats <0.5s)
- Pass rate: Should maintain >90%

### Phase 1B: Cache All LLM Calls
- Repeated queries: <0.3s (98% faster)
- First-time queries: Same as Phase 1
- Pass rate: Should maintain >90%

### Phase 1C: Merge Domain Reasoning → SQL
- All queries: ~6s faster (12% improvement)
- Simple queries: 5s → 4s
- Complex queries: 26s → 20s
- Pass rate: Should maintain >90%

---

## Pro Tips

1. **Save Results:** Archive JSON outputs to track quality over time
2. **Browser Bookmark:** Save evaluation URL with ModHeader for quick access
3. **Monitoring:** Parse JSON and send Slack/Discord notifications
4. **Schedule Checks:** Use cron or GitHub Actions for periodic tests
5. **Baseline First:** Always run baseline before making changes
6. **Test Locally:** Test locally before Railway deployment

---

## Example Workflow

```bash
# 1. Set up environment
export RAILWAY_URL="https://langchain-railway.railway.app"
export API_KEY="sk_test123456789"

# 2. Run quick test
echo "Running quick evaluation..."
curl -H "X-App-Key: $API_KEY" \
  "$RAILWAY_URL/evaluate?mode=quick&format=json" \
  > results.json

# 3. Check pass rate
PASS_RATE=$(jq '.summary.pass_rate * 100' results.json)
echo "Pass rate: $PASS_RATE%"

# 4. If pass rate < 90%, investigate
if (( $(echo "$PASS_RATE < 90" | bc -l) )); then
  echo "WARNING: Pass rate below 90%"
  echo "Failed queries:"
  jq '.results[] | select(.status == "fail") | .id' results.json

  echo "\nFailure reasons:"
  jq '.results[] | select(.status == "fail") | {id, reason}' results.json
else
  echo "✓ All systems operational"
fi

# 5. Check performance
AVG_TIME=$(jq '.summary.average_response_time_ms' results.json)
echo "Average response time: ${AVG_TIME}ms"
```

---

## Configuration

Set environment variables or use `.env` file:
```bash
API_URL=http://localhost:8000/ask
APP_SECRET_KEY=your_secret_key
```

For Railway:
- URL from Railway Dashboard → Settings
- API key from Railway Dashboard → Variables → `APP_SECRET_KEY`

---

## Files Reference

- **evaluation_dataset.json** - 75 test queries with criteria
- **test_evaluation.py** - Automated test runner
- **main.py** - `/evaluate` endpoint implementation (line ~2512)
- **docs/DEVELOPER_GUIDE.md** - Code architecture and best practices
- **docs/CHANGELOG.md** - Version history and optimization notes

---

## Summary

**Quick Workflow:**
1. Get Railway URL and API_KEY
2. Install ModHeader OR use cURL
3. Visit: `https://your-url.railway.app/evaluate?mode=quick`
4. Add header: `X-App-Key: your_key`
5. Wait 1-2 minutes
6. Check pass rate ≥90% ✓
7. Done!

**Target Metrics:**
- Pass rate: ≥90%
- Simple queries: <8s
- Complex queries: <45s
- Cache hit rate: >60%

For detailed architecture and code review, see `docs/DEVELOPER_GUIDE.md`.
