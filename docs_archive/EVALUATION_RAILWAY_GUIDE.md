# Running Evaluations on Railway Production

Since you don't have local access, the evaluation system is now accessible via web endpoint on your Railway deployment.

## 🌐 Access Evaluation Endpoint

Your Railway app now has a `/evaluate` endpoint that runs quality tests and shows results in your browser.

### URL Format

```
https://your-railway-app.railway.app/evaluate?mode=quick
```

Replace `your-railway-app.railway.app` with your actual Railway domain.

---

## 🔐 Authentication

The `/evaluate` endpoint requires your API key in the header:

**Header:** `X-App-Key: your_secret_key_here`

### Using Browser (with extension)

1. Install "ModHeader" or "Simple Modify Headers" browser extension
2. Add header: `X-App-Key` = `your_secret_key`
3. Visit the URL in your browser

### Using cURL

```bash
curl -H "X-App-Key: your_secret_key_here" \
  "https://your-railway-app.railway.app/evaluate?mode=quick"
```

### Using Postman/Insomnia

1. Create new GET request
2. URL: `https://your-railway-app.railway.app/evaluate`
3. Headers: Add `X-App-Key: your_secret_key_here`
4. Params: Add query parameters (see below)
5. Send request

---

## 📊 Usage Examples

### 1. Quick Test (10 queries, ~1-2 minutes)
**Best for:** Quick smoke test after deployment

```
GET /evaluate?mode=quick&format=html
```

Shows pass/fail summary with nice HTML report in browser.

### 2. Full Test (75 queries, ~10-15 minutes)
**Best for:** Comprehensive validation before production release

```
GET /evaluate?mode=full&format=html
```

Tests all query types thoroughly.

### 3. Test Specific Query Type
**Best for:** Debugging specific functionality

```bash
# Test only simple lookups
GET /evaluate?type=single_value

# Test only analytical queries
GET /evaluate?type=analyst

# Test only trend analysis
GET /evaluate?type=trend

# Test only comparisons
GET /evaluate?type=comparison

# Test only list queries
GET /evaluate?type=list
```

### 4. Test Specific Query
**Best for:** Debugging a specific test case

```
GET /evaluate?query_id=sv_001
```

### 5. Get JSON Results
**Best for:** Programmatic access, CI/CD integration

```
GET /evaluate?mode=quick&format=json
```

Returns JSON with full results data:
```json
{
  "summary": {
    "pass_rate": 0.92,
    "total_queries": 10,
    "passed": 9,
    "failed": 1,
    "performance": {...},
    "by_type": {...}
  },
  "results": [...]
}
```

---

## 📈 Understanding Results

### HTML Report (Browser View)

The HTML report shows:

**Pass Rate** - Overall percentage of tests passing
- **Green (≥90%)**: Production ready ✓
- **Orange (70-89%)**: Investigate issues
- **Red (<70%)**: Critical problems

**Performance Metrics**
- Avg Response Time: Overall average
- Simple Queries: Should be <8s
- Complex Queries: Should be <45s

**Results by Type**
- Pass rate for each query type
- Helps identify which functionality has issues

**Detailed Results Table**
- Each test with pass/fail status
- Execution time
- Specific issues (SQL, quality, performance)

### JSON Response

For programmatic access:
```bash
curl -H "X-App-Key: your_key" \
  "https://your-app.railway.app/evaluate?mode=full&format=json" \
  > results.json
```

Then analyze with jq:
```bash
# Get pass rate
jq '.summary.pass_rate' results.json

# Get failed queries
jq '.results[] | select(.status == "fail") | .id' results.json

# Get avg performance
jq '.summary.performance.avg_time_ms' results.json
```

---

## 🔄 Recommended Testing Schedule

### After Each Deployment
```
GET /evaluate?mode=quick&format=html
```
**Expected time:** 1-2 minutes
**Action:** Verify pass rate ≥90%

### Before Major Release
```
GET /evaluate?mode=full&format=html
```
**Expected time:** 10-15 minutes
**Action:** Review all failures, ensure no regressions

### Performance Monitoring
```
GET /evaluate?type=single_value&format=json
GET /evaluate?type=analyst&format=json
```
**Action:** Track response times over time

---

## ⚠️ Important Notes

### Rate Limiting
The `/evaluate` endpoint bypasses the normal 10/minute rate limit because it needs to run many queries. However, don't run multiple evaluations simultaneously.

### Timeout
Full evaluations with 75 queries may take 10-15 minutes. Make sure your HTTP client doesn't timeout:
- cURL: Add `--max-time 1200` (20 minutes)
- Postman: Increase timeout in Settings
- Browser: Should handle automatically

### Database Load
Running evaluations will:
- Execute 10-75 SQL queries
- Generate 10-75 LLM calls
- May be slow if cache is cold

**Recommendation:** Run during low-traffic periods or use `mode=quick` for routine checks.

---

## 🐛 Troubleshooting

### Error: "Evaluation dataset not found"
**Cause:** `evaluation_dataset.json` not deployed
**Fix:** Ensure all files are committed and deployed to Railway

### Error: 401 Unauthorized
**Cause:** Missing or wrong API key
**Fix:** Check `X-App-Key` header matches your `APP_SECRET_KEY` env var

### Error: 503 Service Unavailable
**Cause:** Server overloaded or database connection issue
**Fix:** Check Railway logs, ensure database is healthy

### Long Response Time
**Cause:** Cache is cold, LLM calls are slow
**Expected:** First run may be slow (10-15 min for full test)
**Fix:** Wait for completion. Subsequent runs will be faster due to caching.

### Low Pass Rate (<70%)
**Causes:**
1. Recent code changes broke functionality
2. Database schema changed
3. LLM model behavior changed
4. Domain knowledge outdated

**Action:**
1. Check Railway logs for errors
2. Run `GET /evaluate?format=json` to get detailed failure info
3. Test specific failing queries: `GET /evaluate?query_id=sv_001`
4. Review recent commits

---

## 📊 Integration with Railway Dashboard

### View Logs
Railway Dashboard → Your App → Logs

Filter for:
- `🧪 Evaluation` - Evaluation endpoint activity
- `✓ PASSED` - Successful tests
- `✗ FAILED` - Failed tests
- `⚡ SQL executed` - Query execution

### Monitor Metrics
Railway Dashboard → Your App → Metrics

Watch for:
- CPU usage spikes during evaluation
- Memory usage
- Response time

---

## 🚀 Quick Start Checklist

1. ✓ Deploy latest code to Railway
2. ✓ Ensure `evaluation_dataset.json` and `evaluation_engine.py` are deployed
3. ✓ Get your Railway app URL from Railway dashboard
4. ✓ Get your `APP_SECRET_KEY` from Railway environment variables
5. ✓ Open browser with header extension OR use cURL
6. ✓ Visit: `https://your-app.railway.app/evaluate?mode=quick`
7. ✓ Add header: `X-App-Key: your_secret`
8. ✓ Wait 1-2 minutes for results
9. ✓ Check pass rate is ≥90%

---

## 💡 Pro Tips

### 1. Save Results Over Time
```bash
# Run and save results with timestamp
curl -H "X-App-Key: $KEY" \
  "https://your-app.railway.app/evaluate?mode=full&format=json" \
  > "results_$(date +%Y%m%d_%H%M%S).json"
```

### 2. Compare Before/After Deployment
```bash
# Before deployment
curl ... > results_before.json

# After deployment
curl ... > results_after.json

# Compare pass rates
diff <(jq .summary.pass_rate results_before.json) \
     <(jq .summary.pass_rate results_after.json)
```

### 3. Focus on Problematic Query Types
```bash
# If analyst queries are failing
curl -H "X-App-Key: $KEY" \
  "https://your-app.railway.app/evaluate?type=analyst&format=json" \
  | jq '.results[] | select(.status == "fail")'
```

### 4. Monitor Cache Effectiveness
```bash
# Run twice and compare times
curl -H "X-App-Key: $KEY" "https://your-app.railway.app/evaluate?mode=quick&format=json" > run1.json
curl -H "X-App-Key: $KEY" "https://your-app.railway.app/evaluate?mode=quick&format=json" > run2.json

# Second run should be much faster due to caching
jq '.summary.performance.avg_time_ms' run1.json
jq '.summary.performance.avg_time_ms' run2.json
```

---

## 📞 Support

If evaluation results show issues:

1. **Check logs** in Railway dashboard for specific errors
2. **Run specific failing query** with `?query_id=xxx` to debug
3. **Review recent commits** for breaking changes
4. **Check database** health via `/metrics` endpoint

For questions about the evaluation system:
- Review `EVALUATION_GUIDE.md` for detailed methodology
- Review `evaluation_dataset.json` for test query definitions
- Check `main.py` line ~2512 for `/evaluate` endpoint implementation
