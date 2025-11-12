# 🚀 Quick Start: Run Evaluations on Railway

## Step 1: Get Your Credentials

You need two things:
1. **Railway URL**: Find in Railway Dashboard → Your App → Settings
   - Example: `https://langchain-railway-production.up.railway.app`

2. **API Key**: Find in Railway Dashboard → Your App → Variables
   - Look for: `APP_SECRET_KEY`
   - Example: `sk_xxxxxxxxxxxxx`

---

## Step 2: Choose Your Method

### Option A: Browser (Easiest)

**Install Browser Extension** (one-time setup):
- Chrome/Edge: Install [ModHeader](https://chrome.google.com/webstore/detail/modheader/idgpnmonknjnojddfkpgkljpfnnfcklj)
- Firefox: Install [Modify Header Value](https://addons.mozilla.org/en-US/firefox/addon/modify-header-value/)

**Configure Header**:
1. Open the extension
2. Add new header:
   - Name: `X-App-Key`
   - Value: `your_api_key_here`

**Run Evaluation**:
1. Visit in browser:
   ```
   https://your-railway-url.railway.app/evaluate?mode=quick
   ```

2. You'll see a nice HTML report with:
   - ✓ Pass rate
   - ✓ Performance metrics
   - ✓ Detailed results

**Expected time:** 1-2 minutes for quick mode

---

### Option B: cURL (Command Line)

```bash
# Set your credentials
export RAILWAY_URL="https://your-app.railway.app"
export API_KEY="your_api_key_here"

# Run quick evaluation (10 queries)
curl -H "X-App-Key: $API_KEY" \
  "$RAILWAY_URL/evaluate?mode=quick&format=json" \
  | jq '.'

# Run full evaluation (75 queries)
curl -H "X-App-Key: $API_KEY" \
  "$RAILWAY_URL/evaluate?mode=full&format=json" \
  | jq '.summary'
```

---

### Option C: Postman/Insomnia (GUI)

**Setup:**
1. Create new GET request
2. URL: `https://your-railway-url.railway.app/evaluate`
3. Headers tab → Add:
   - Key: `X-App-Key`
   - Value: `your_api_key`
4. Params tab → Add:
   - Key: `mode`
   - Value: `quick`
   - Key: `format`
   - Value: `json`
5. Click "Send"

---

## Step 3: Interpret Results

### Pass Rate (Most Important)

```
Pass Rate: 92.5%  ← This number!
```

**What it means:**
- **≥90% (Green)**: ✅ Production ready - all systems working well
- **70-89% (Orange)**: ⚠️ Some issues - review failures before deploying
- **<70% (Red)**: ❌ Critical problems - do not deploy

### Performance

```
Simple Queries: 3200ms (target: <8s)  ✅ Good
Complex Queries: 18500ms (target: <45s)  ✅ Good
```

**What it means:**
- Simple queries should be fast (<8 seconds)
- Complex queries can take longer (<45 seconds)
- If exceeded, check Railway logs

### By Query Type

```
single_value   : 22/23 passed (95.7%)  ← Simple lookups
analyst        :  6/ 7 passed (85.7%)  ← Complex analysis
```

**What it means:**
- Shows which features are working
- single_value failures = basic functionality broken
- analyst failures = complex reasoning issues

---

## Common Commands

### Quick Smoke Test (After Deployment)
```bash
curl -H "X-App-Key: $API_KEY" \
  "$RAILWAY_URL/evaluate?mode=quick"
```
**Time:** 1-2 minutes
**Use:** After every deployment to verify nothing broke

### Full Test (Before Production Release)
```bash
curl -H "X-App-Key: $API_KEY" \
  "$RAILWAY_URL/evaluate?mode=full&format=json" \
  > results.json
```
**Time:** 10-15 minutes
**Use:** Before major releases, comprehensive validation

### Test Specific Feature
```bash
# Test only analytical queries
curl -H "X-App-Key: $API_KEY" \
  "$RAILWAY_URL/evaluate?type=analyst"

# Test only simple lookups
curl -H "X-App-Key: $API_KEY" \
  "$RAILWAY_URL/evaluate?type=single_value"
```
**Use:** When specific feature has issues

### Get JSON for Automation
```bash
curl -H "X-App-Key: $API_KEY" \
  "$RAILWAY_URL/evaluate?mode=quick&format=json" \
  | jq '.summary.pass_rate'
```
**Use:** CI/CD integration, monitoring

---

## Troubleshooting

### "401 Unauthorized"
**Problem:** API key is wrong or missing
**Fix:** Check `X-App-Key` header matches Railway `APP_SECRET_KEY` env variable

### "503 Evaluation dataset not found"
**Problem:** Files not deployed to Railway
**Fix:**
1. Ensure you've pushed latest code
2. Railway should auto-deploy
3. Check Railway logs for deployment errors

### Request Times Out
**Problem:** Evaluation takes too long (10-15 min for full test)
**Fix:** Use `mode=quick` instead, or increase timeout:
```bash
curl --max-time 1200 ...  # 20 minute timeout
```

### Low Pass Rate (<70%)
**Problem:** Recent changes broke functionality
**Fix:**
1. Check Railway logs for errors
2. Run specific failing test: `?query_id=sv_001`
3. Review recent commits
4. Consider rollback

---

## What to Monitor

### After Each Deployment
✅ Run: `?mode=quick` (1-2 min)
✅ Check: Pass rate ≥90%
✅ Check: Performance within targets

### Weekly/Monthly
✅ Run: `?mode=full` (10-15 min)
✅ Compare: Pass rate trend over time
✅ Save results: Archive JSON for history

---

## Example Session

```bash
# 1. Set credentials
export RAILWAY_URL="https://langchain-railway-production.up.railway.app"
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
  jq '.results[] | select(.status == "fail") | .id' results.json
else
  echo "✓ All systems operational"
fi
```

---

## Need More Help?

**Detailed Guide:** See `EVALUATION_RAILWAY_GUIDE.md`
**Dataset Details:** See `evaluation_dataset.json`
**Endpoint Code:** See `main.py` line ~2512

**Railway Dashboard:**
- Logs: Track evaluation progress
- Metrics: Monitor resource usage
- Variables: Verify API_KEY is set

---

## Pro Tips

1. **Save Results:** Archive JSON outputs to track quality over time
2. **Browser Bookmark:** Save the evaluation URL with header extension for quick access
3. **Slack/Discord Integration:** Parse JSON and send pass/fail notifications
4. **Schedule Checks:** Use cron or GitHub Actions to run periodic evaluations

---

## Summary

**Quick workflow:**
1. Get Railway URL and API_KEY
2. Install ModHeader browser extension OR use cURL
3. Visit: `https://your-url.railway.app/evaluate?mode=quick`
4. Add header: `X-App-Key: your_key`
5. Wait 1-2 minutes
6. Check pass rate ≥90% ✓
7. Done!

**Questions?** Review `EVALUATION_RAILWAY_GUIDE.md` for complete documentation.
