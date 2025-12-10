# Energy Chatbot - Developer Guide

Complete guide for developers working on the energy chatbot.

---

## Architecture Overview

### Technology Stack
- **Backend:** FastAPI (Python 3.11+)
- **LLM:** Google Gemini 2.5 Flash (primary), GPT-4o-mini (fallback)
- **Database:** Supabase PostgreSQL (materialized views)
- **LangChain:** Query orchestration, prompt management
- **Deployment:** Railway

### Current Version
**v18.7 (Gemini Analyst)**
- Merged domain reasoning into SQL generation
- Conditional domain guidance for performance
- LLM response caching (in-memory)
- Read-only SQL enforcement
- Rate limiting (10 req/min per IP)

---

## Project Structure

```
langchain_railway/
├── main.py                       # FastAPI app (3,900 lines - NEEDS REFACTORING)
├── context.py                    # Schema, column labels, classifications
├── domain_knowledge.py           # Energy market domain facts
├── evaluation_engine.py          # Test harness
├── test_evaluation.py            # Automated test runner
├── evaluation_dataset.json       # 75 test queries
├── requirements.txt              # Dependencies
├── .env                          # Environment variables (not in git)
├── docs/
│   ├── EVALUATION.md             # Testing guide
│   ├── DEVELOPER_GUIDE.md        # This file
│   └── CHANGELOG.md              # Version history
└── tests/                        # Unit tests (minimal coverage)
    ├── test_api.py
    ├── test_sql.py
    └── test_shares.py
```

**⚠️ ISSUE:** main.py is monolithic (3,900 lines). See COMPREHENSIVE_AUDIT.md for refactoring plan.

---

## Core Components

### 1. Request Processing Pipeline

```
User Query
    ↓
1. Language Detection (detect_language)
    ↓
2. Analysis Mode Detection (light/analyst)
    ↓
3. LLM: Generate Plan + SQL (llm_generate_plan_and_sql)
    ↓
4. SQL Validation & Repair (validate_and_fix_sql)
    ↓
5. SQL Execution (READ ONLY, 30s timeout)
    ↓
6. Data Filtering (supply/demand/transit)
    ↓
7. Quick Statistics (trends, CAGR, seasonal)
    ↓
8. LLM: Summarization (llm_summarize)
    ↓
9. Chart Generation (semantic chart type selection)
    ↓
10. APIResponse (answer, chart_data, chart_type, execution_time)
```

### 2. LLM Functions

#### `llm_generate_plan_and_sql()`
**Location:** main.py:1829-1968

**Purpose:** Generate analysis plan + SQL query in ONE call (Phase 1C optimization)

**Input:**
- User query
- Analysis mode (light/analyst)
- Language instruction
- Domain knowledge (selective, based on query focus)

**Output:**
```
{JSON plan}
---SQL---
SELECT ...
```

**Optimizations:**
- LLM response caching (SHA256 hash of prompt)
- Selective domain knowledge (30-40% token reduction)
- Conditional guidance (balancing/tariff/CPI only when needed)

**Few-Shot Examples:** 13 SQL examples (main.py:1550-1825)

#### `llm_summarize()`
**Location:** main.py:2129-2320

**Purpose:** Generate final answer text from data

**Input:**
- User query
- Data preview (max 200 rows)
- Stats hint (trends, CAGR, seasonal patterns)
- Language instruction
- Domain knowledge (conditional)

**Output:** Natural language answer

**Optimizations:**
- Simple queries skip domain knowledge
- Complex queries get full guidance
- Georgian/Russian language support

#### `llm_analyze_with_domain_knowledge()`
**Location:** main.py:1100-1144

**Status:** DEPRECATED (merged into llm_generate_plan_and_sql in Phase 1C)

### 3. SQL Generation & Validation

#### Table Whitelist
```python
ALLOWED_TABLES = {
    "entities_mv",           # Power sector entities
    "price_with_usd",        # Market prices (GEL/USD)
    "tariff_with_usd",       # Regulated tariffs
    "tech_quantity_view",    # Generation by technology
    "trade_derived_entities",# Trading volumes
    "monthly_cpi_mv",        # CPI inflation
    "energy_balance_long_mv" # National energy balance
}
```

#### Validation Steps
```python
def validate_and_fix_sql(raw_sql: str) -> str:
    """
    1. Parse SQL with SQLGlot (AST parsing)
    2. Extract table references (including CTEs)
    3. Check against ALLOWED_TABLES whitelist
    4. Apply synonym auto-correction
       - prices → price_with_usd
       - tariffs → tariff_with_usd
       - tech_quantity → tech_quantity_view
    5. Return validated SQL or raise error
    """
```

**Security:**
- `SET TRANSACTION READ ONLY` on every execution
- No INSERT/UPDATE/DELETE allowed
- 30-second statement timeout
- AST-based whitelist (not regex)

### 4. Chart Generation

**Location:** main.py:3466-3700

**Semantic-Aware Chart Type Selection:**

```python
# Detection flow:
1. Identify time columns (year, month, date)
2. Identify category columns (type, entity, sector)
3. Identify value columns (quantity, price, tariff)
4. Infer dimensions (xrate, share, price_tariff, energy_qty, index)
5. Apply decision matrix
```

**Decision Matrix:**

| Structure | Dimension | Chart Type | Example |
|-----------|-----------|------------|---------|
| Time + Share | share | stackedbar | Composition over time |
| Time + No Category | price/qty | line | Single time series |
| Time + Categories | price/qty | line | Multi-line trend |
| No Time + Share (≤8 cats) | share | pie | Composition snapshot |
| No Time + Categories | any | bar | Categorical comparison |

**⚠️ ISSUE:** Chart and answer generated separately → can mismatch. See COMPREHENSIVE_AUDIT.md Section 3.

### 5. Data Processing Functions

#### `build_balancing_correlation_df()`
**Location:** main.py:677-752

**Purpose:** Build monthly decomposition of balancing price with entity shares

**Calculation:**
```python
# For each month:
share_import = qty_import / total_balancing_qty
share_deregulated_hydro = qty_dereg_hydro / total_balancing_qty
share_regulated_hpp = qty_reg_hpp / total_balancing_qty
# ... 7 more shares

# IMPORTANT: Only uses segment='balancing_electricity'
```

**Returns:**
```python
DataFrame with columns:
- date, p_bal_gel, p_bal_usd, xrate
- share_import, share_deregulated_hydro, share_regulated_hpp, ...
- enguri_tariff_gel, gardabani_tpp_tariff_gel, ...
```

#### `compute_seasonal_average()`
**Location:** main.py:797-821

**Purpose:** Decompose price/quantity by season

**Seasons:**
- **Summer:** April-July (months 4,5,6,7) - Hydro dominant, low prices
- **Winter:** August-March - Thermal/import dominant, high prices

**Aggregation:**
- SUM for quantities
- AVG for prices
- Separate CAGR for each season

#### `quick_stats()`
**Location:** main.py:1984-2126

**Purpose:** Auto-generate statistics hint for LLM

**Detects:**
- Date columns → trends across first/last years
- Percentage change: `((last - first) / first * 100)`
- Seasonal CAGR for price columns
- Numeric ranges, counts

---

## Database Schema

### Materialized Views

#### `entities_mv`
```sql
entity           -- Full entity name
entity_normalized-- Standardized ID
type             -- HPP, TPP, etc.
ownership        -- State, private
source           -- Local, import-dependent
```

#### `price_with_usd`
```sql
date             -- YYYY-MM-DD
p_dereg_gel      -- Deregulated price (GEL/MWh)
p_bal_gel        -- Balancing electricity price (GEL/MWh)
p_gcap_gel       -- Guaranteed capacity fee (GEL/MWh)
xrate            -- Exchange rate (GEL/USD)
p_dereg_usd      -- = p_dereg_gel / xrate
p_bal_usd        -- = p_bal_gel / xrate
p_gcap_usd       -- = p_gcap_gel / xrate
```

#### `tariff_with_usd`
```sql
date             -- YYYY-MM-DD
entity           -- Generator entity
tariff_gel       -- Regulated tariff (GEL/MWh)
tariff_usd       -- = tariff_gel / xrate
```

#### `tech_quantity_view`
```sql
date             -- YYYY-MM-DD
type_tech        -- hydro, thermal, wind, solar, import, export, etc.
quantity_tech    -- Quantity (thousand MWh) - MULTIPLY BY 1000 FOR MWh
```

**Type_tech classification:**
- **Supply:** hydro, thermal, wind, solar, import, self-cons
- **Demand:** abkhazeti, supply-distribution, direct customers, losses, export
- **Transit:** transit

#### `trade_derived_entities`
```sql
date             -- YYYY-MM-DD
entity           -- Trading entity (deregulated_hydro, import, regulated_hpp, etc.)
segment          -- balancing_electricity, bilateral_exchange
quantity         -- Trade volume (thousand MWh)
```

**Key Entities:**
- deregulated_hydro
- import
- regulated_hpp
- regulated_new_tpp
- regulated_old_tpp
- renewable_ppa
- thermal_ppa

### Joins

```python
DB_JOINS = {
    "price_with_usd": {
        "join_on": "date",
        "related_to": ["tariff_with_usd", "tech_quantity_view", "trade_derived_entities"]
    },
    "tariff_with_usd": {
        "join_on": ["date", "entity"],
        "related_to": ["price_with_usd", "trade_derived_entities"]
    },
    # ...
}
```

---

## Domain Knowledge

**File:** `domain_knowledge.py`

**Key Facts:**

### Balancing Price Formation
- **Weighted average** of electricity sold on balancing market
- **Primary Driver #1:** Exchange rate (xrate) - MOST IMPORTANT for GEL prices
- **Primary Driver #2:** Composition (entity shares) - CRITICAL for all prices

**Price Hierarchy (cheapest → most expensive):**
1. Regulated HPP: ~20-30 GEL/MWh
2. Deregulated Hydro: ~30-50 GEL/MWh
3. Regulated Thermal: cost-plus, varies with gas price
4. Renewable PPA: CONFIDENTIAL (~57-60 $/MWh, internal only)
5. Thermal PPA: CONFIDENTIAL
6. Import: CONFIDENTIAL (usually highest)

### Seasonality
- **Summer (Apr-Jul):** Hydro dominant (>60%), low prices
- **Winter (Aug-Mar):** Thermal/import dominant (<30% hydro), high prices

### CfD Contracts
- Contracts for Difference for new renewable projects
- **Centrally dispatched** by GSE (not on exchange)
- **Price-neutral** (fixed strike price)
- **Quantity risk** (curtailment not compensated)

### Balancing Market Structure
- **Current:** Monthly imbalance settlement (NOT real-time)
- **Future (2027):** Hourly balancing with BRPs

### Confidentiality Rules
- **DO disclose:** Regulated tariffs, deregulated hydro prices, xrate
- **DO NOT disclose:** PPA price estimates, import prices
- When discussing PPA/import: say "varies" or "market-based"

---

## Code Quality Issues

See `COMPREHENSIVE_AUDIT.md` for full details. Key issues:

### CRITICAL Issues

#### 1. Unreliable SQL Generation for Aggregations
**Problem:** LLM might not generate correct SUM/GROUP BY for totals

**Example Failure:**
```
User: "What was total generation in 2023?"

LLM generates:
SELECT type_tech, quantity_tech FROM tech_quantity_view WHERE ...
❌ WRONG - doesn't SUM across technologies

Should be:
SELECT SUM(quantity_tech) as total FROM tech_quantity_view WHERE ...
✅ CORRECT
```

**Root Cause:**
- Only 1 of 13 few-shot examples shows total calculation
- No validation that SQL matches aggregation intent

**Fix:** See COMPREHENSIVE_AUDIT.md Section 2

#### 2. Chart-Answer Mismatch
**Problem:** Chart generated AFTER answer, no feedback loop

**Example:**
```
Answer: "Price increased due to import share and exchange rate"
Chart: Only shows price over time
Missing: import share, exchange rate
```

**Fix:** See COMPREHENSIVE_AUDIT.md Section 3

### HIGH Priority Issues

#### 3. Monolithic Architecture
**Problem:** main.py = 3,900 lines, hard to maintain

**Recommended Structure:**
```
core/
  ├── llm.py              # LLM chains
  ├── sql_generator.py    # SQL generation + validation
  ├── query_executor.py   # DB execution
  └── cache.py            # Caching logic
analysis/
  ├── shares.py           # Share calculations
  ├── seasonal.py         # Seasonal decomposition
  └── stats.py            # Statistics
visualization/
  ├── chart_selector.py   # Chart type logic
  └── chart_builder.py    # Data formatting
```

**Benefits:**
- Each module <500 lines
- Easy to unit test
- Clear separation of concerns

---

## Best Practices

### SQL Generation

**DO:**
- Use materialized views only (never raw tables)
- Add few-shot examples for new query types
- Validate SQL matches user intent
- Use English column aliases (even for Georgian/Russian queries)

**DON'T:**
- Access tables not in whitelist
- Generate INSERT/UPDATE/DELETE
- Use comments in SQL (stripped by validator)
- Expose PPA/import pricing

### LLM Prompts

**DO:**
- Cache prompts with SHA256 hash
- Use selective domain knowledge (query-specific)
- Provide clear examples
- Specify output format

**DON'T:**
- Include entire domain knowledge for simple queries
- Hallucinate column names
- Generate SQL without schema context

### Chart Generation

**DO:**
- Match chart to answer content
- Use semantic dimension detection
- Apply human-readable labels
- Limit to 4-5 key indicators

**DON'T:**
- Show all columns (overwhelms user)
- Mix incompatible units without dual axes
- Generate charts for explanatory queries

### Testing

**DO:**
- Run `mode=quick` after every change
- Run `mode=full` before deployment
- Add test cases for new features
- Track pass rate over time

**DON'T:**
- Deploy if pass rate <90%
- Skip testing on "small changes"
- Ignore performance regressions

---

## Environment Variables

```bash
# Required
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your_key
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key  # Fallback
APP_SECRET_KEY=your_secret      # API authentication

# Optional
MODEL_TYPE=gemini               # or 'openai'
PORT=8000                       # Default FastAPI port
ENVIRONMENT=production          # or 'development'
```

---

## Development Workflow

### Local Setup

```bash
# 1. Clone repo
git clone <repo_url>
cd langchain_railway

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up .env
cp .env.example .env
# Edit .env with your credentials

# 5. Run server
uvicorn main:app --reload --port 8000

# 6. Test
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-App-Key: your_key" \
  -d '{"query": "What was balancing price in June 2024?"}'
```

### Making Changes

```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes
# Edit code...

# 3. Run tests locally
python test_evaluation.py --mode quick

# 4. Check pass rate ≥90%
# If failed, fix issues

# 5. Commit
git add .
git commit -m "feat: your feature description"

# 6. Push
git push origin feature/your-feature

# 7. Railway auto-deploys
# Wait for deployment

# 8. Test on Railway
curl -H "X-App-Key: $API_KEY" \
  "$RAILWAY_URL/evaluate?mode=quick"

# 9. If pass rate ≥90%, merge to main
```

### Debugging

```bash
# Check logs locally
uvicorn main:app --log-level debug

# Check Railway logs
railway logs --follow

# Test specific query
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-App-Key: your_key" \
  -d '{"query": "your test query", "mode": "analyst"}' \
  | jq '.'

# Check metrics
curl http://localhost:8000/metrics | jq '.'
```

---

## Performance Optimization

### Current Optimizations

#### Phase 1: Response Caching
- In-memory cache (max 1000 entries)
- SHA256 hash of prompts
- 50-70% hit rate expected
- Simple queries: 26s → 3-5s

#### Phase 1B: Cache All LLM Calls
- Cache SQL generation, summarization
- Repeated queries: <0.3s

#### Phase 1C: Merge Domain Reasoning
- Combine domain analysis + SQL generation
- All queries: ~6s faster (12% improvement)

#### Phase 1D: Security Hardening
- Read-only enforcement
- Rate limiting (10 req/min per IP)
- 30-second query timeout

### Future Optimizations

- **Persistent cache:** Redis instead of in-memory
- **Materialized shares view:** Move Python calculations to DB
- **Query plan caching:** PostgreSQL prepared statements
- **Async processing:** Background chart generation
- **CDN:** Cache static responses

---

## Deployment

### Railway

**Auto-deployment:** Push to `main` branch

**Environment:**
- Set variables in Railway Dashboard → Variables
- Check logs: Railway Dashboard → Logs
- Monitor metrics: Railway Dashboard → Metrics

**Health Check:**
```bash
curl $RAILWAY_URL/metrics
```

### Manual Deployment

```bash
# 1. Build
docker build -t energy-chatbot .

# 2. Run
docker run -p 8000:8000 \
  -e SUPABASE_URL=$SUPABASE_URL \
  -e SUPABASE_KEY=$SUPABASE_KEY \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e APP_SECRET_KEY=$APP_SECRET_KEY \
  energy-chatbot

# 3. Test
curl http://localhost:8000/metrics
```

---

## Monitoring

### Metrics Endpoint

```bash
GET /metrics
```

**Returns:**
```json
{
  "total_requests": 1234,
  "total_llm_calls": 567,
  "total_sql_executions": 543,
  "total_errors": 12,
  "cache_hit_rate": 0.68,
  "average_response_time": 4.2
}
```

### What to Monitor

- **Pass rate:** Should stay ≥90%
- **Response time:** Simple <8s, Complex <45s
- **Cache hit rate:** Should be >60%
- **Error rate:** Should be <5%
- **SQL execution time:** Should be <2s

---

## Troubleshooting

### Common Issues

#### LLM Errors
**Symptoms:** 500 errors, "LLM call failed"
**Causes:**
- API key expired/invalid
- Rate limiting
- Network timeout

**Fix:**
- Check API keys in .env
- Review LLM API status
- Increase timeout if needed

#### SQL Validation Errors
**Symptoms:** "SQL validation failed", "Table not in whitelist"
**Causes:**
- LLM generated invalid table name
- Synonym not in mapping

**Fix:**
- Add synonym to `SYNONYM_MAP` in main.py
- Add few-shot example for query type
- Review prompt guidance

#### Chart Generation Issues
**Symptoms:** No chart, wrong chart type, missing variables
**Causes:**
- Chart necessity detection too strict
- Dimension detection failed
- Data structure unexpected

**Fix:**
- Review `should_generate_chart()` logic
- Check dimension inference in `infer_dimension()`
- Add debug logs for chart selection

---

## Testing

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_sql.py::test_aggregation_logic

# With coverage
pytest --cov=main tests/
```

### Integration Tests
```bash
# Run evaluation suite
python test_evaluation.py --mode full
```

### Manual Testing
```bash
# Test specific query
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-App-Key: test_key" \
  -d '{
    "query": "What was balancing price in June 2024?",
    "mode": "light"
  }' | jq '.'
```

---

## Contributing

### Code Style
- PEP 8 compliant
- Type hints preferred
- Docstrings for public functions
- Comments for complex logic

### Commit Messages
```
feat: Add aggregation intent detection
fix: Correct share calculation for renewable PPA
docs: Update evaluation guide
perf: Improve LLM caching hit rate
test: Add unit tests for SQL generation
```

### Pull Request Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] `mode=quick` passes locally
- [ ] Documentation updated
- [ ] No secrets in code
- [ ] Performance impact assessed

---

## Resources

- **Evaluation Guide:** docs/EVALUATION.md
- **Changelog:** docs/CHANGELOG.md
- **Audit Report:** COMPREHENSIVE_AUDIT.md
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **LangChain Docs:** https://python.langchain.com
- **Supabase Docs:** https://supabase.com/docs

---

## Support

For questions or issues:
1. Check this guide first
2. Review COMPREHENSIVE_AUDIT.md for known issues
3. Check Railway logs for errors
4. Review /metrics endpoint
5. Run evaluation tests to isolate issue

**Common Questions:**
- "Why is my query slow?" → Check /metrics, review LLM cache hit rate
- "Why is SQL wrong?" → Check few-shot examples, add similar query
- "Why is chart missing variables?" → See COMPREHENSIVE_AUDIT Section 3
- "Why is pass rate <90%?" → Run `mode=full`, review failures

---

## Next Steps

After reading this guide:

1. **Setup:** Follow local setup instructions
2. **Explore:** Read main.py functions (start with ask_post())
3. **Test:** Run `python test_evaluation.py --mode quick`
4. **Fix:** Review COMPREHENSIVE_AUDIT.md for priority issues
5. **Improve:** Pick an issue and create PR

**Priority areas for improvement:**
1. Add aggregation intent detection (CRITICAL)
2. Link chart to answer content (CRITICAL)
3. Refactor main.py into modules (HIGH)
4. Add unit tests for calculations (HIGH)
5. Create materialized shares view (MEDIUM)

See COMPREHENSIVE_AUDIT.md Section 7 for full action items.
