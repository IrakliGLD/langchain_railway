# Energy Chatbot for Georgian Electricity Market

AI-powered chatbot for analyzing Georgian electricity market data using natural language queries in English, Georgian, and Russian.

## Features

- **Multi-language Support:** English, Georgian, Russian
- **Natural Language to SQL:** Automatic query generation from user questions
- **Smart Chart Generation:** Semantic chart type selection based on data structure
- **Domain Knowledge Integration:** Comprehensive understanding of Georgian energy market
- **Analysis Modes:**
  - **Light Mode:** Quick answers for simple queries
  - **Analyst Mode:** Deep analysis with price drivers and correlations

## Quick Start

### Prerequisites
- Python 3.11+
- Supabase account (database)
- Google Gemini API key (or OpenAI API key)

### Installation

```bash
# Clone repository
git clone <repo_url>
cd langchain_railway

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials:
#   SUPABASE_URL=your_supabase_url
#   SUPABASE_KEY=your_supabase_key
#   GEMINI_API_KEY=your_gemini_key
#   APP_SECRET_KEY=your_secret_key
```

### Run Locally

```bash
# Start server
uvicorn main:app --reload --port 8000

# Test with cURL
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-App-Key: your_secret_key" \
  -d '{"query": "What was balancing price in June 2024?", "mode": "light"}'
```

### Run Tests

```bash
# Quick test (10 queries, 1-2 minutes)
python test_evaluation.py --mode quick

# Full test (75 queries, 10-15 minutes)
python test_evaluation.py --mode full

# Test specific type
python test_evaluation.py --type analyst
```

## API Endpoints

### POST /ask
Main query endpoint

**Request:**
```json
{
  "query": "What was balancing price in June 2024?",
  "mode": "light"  // or "analyst"
}
```

**Headers:**
```
Content-Type: application/json
X-App-Key: your_secret_key
```

**Response:**
```json
{
  "answer": "The balancing electricity price in June 2024 was 45.2 GEL/MWh...",
  "chart_data": [...],
  "chart_type": "line",
  "chart_metadata": {...},
  "execution_time": 4.2
}
```

### GET /metrics
Application metrics

**Response:**
```json
{
  "total_requests": 1234,
  "total_llm_calls": 567,
  "cache_hit_rate": 0.68,
  "average_response_time": 4.2
}
```

### GET /evaluate
Quality evaluation endpoint

**Parameters:**
- `mode`: "quick" (10 queries) or "full" (75 queries)
- `type`: "single_value", "list", "comparison", "trend", "analyst"
- `format`: "json" or "html"

**Example:**
```bash
curl -H "X-App-Key: your_key" \
  "http://localhost:8000/evaluate?mode=quick&format=json"
```

## Example Queries

### English
- "What was balancing price in June 2024?"
- "Show me hydro generation trend for 2023"
- "Compare regulated vs deregulated tariffs"
- "Why did balancing price increase in winter 2024?"

### Georgian
- "რა იყო საბალანსო ფასი ივნისში 2024?"
- "მაჩვენე ჰიდრო გენერაციის ტრენდი 2023-ში"
- "შეადარე რეგულირებული და დერეგულირებული ტარიფები"

### Russian
- "Какая была балансовая цена в июне 2024?"
- "Покажи тренд гидрогенерации за 2023 год"
- "Сравни регулируемые и дерегулированные тарифы"

## Architecture

### Processing Pipeline
```
User Query
  ↓
1. Language Detection
  ↓
2. Analysis Mode Detection
  ↓
3. LLM: Generate Plan + SQL
  ↓
4. SQL Validation & Security Check
  ↓
5. SQL Execution (Read-Only)
  ↓
6. Data Processing & Statistics
  ↓
7. LLM: Answer Generation
  ↓
8. Chart Generation (Semantic Type Selection)
  ↓
Response (Answer + Chart + Metadata)
```

### Database Schema
- `entities_mv` - Power sector entities
- `price_with_usd` - Market prices (GEL/USD)
- `tariff_with_usd` - Regulated tariffs
- `tech_quantity_view` - Generation by technology
- `trade_derived_entities` - Trading volumes

## Performance

### Response Times
- **Simple queries:** <5s (target), 3-5s (typical)
- **Complex queries:** <20s (target), 15-20s (typical)
- **Cached queries:** <0.5s

### Quality Metrics
- **Overall pass rate:** 92% (target: ≥90%)
- **Simple queries:** 95-100% accuracy
- **Analyst queries:** 70-90% accuracy
- **Cache hit rate:** 60-70%

## Documentation

### For Users
- **Quick Start:** See above
- **API Reference:** Check `/docs` endpoint (FastAPI auto-docs)

### For Developers
- **Developer Guide:** `docs/DEVELOPER_GUIDE.md` - Architecture, code structure, best practices
- **Evaluation Guide:** `docs/EVALUATION.md` - Testing and quality validation
- **Changelog:** `docs/CHANGELOG.md` - Version history and optimizations
- **Code Audit:** `COMPREHENSIVE_AUDIT.md` - Current issues and improvement plan

### Archived Docs
- `docs_archive/` - Old documentation files (reference only)

## Development Workflow

### Making Changes
```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes
# Edit code...

# 3. Run tests
python test_evaluation.py --mode quick

# 4. Check pass rate ≥90%
# If failed, fix issues

# 5. Commit and push
git add .
git commit -m "feat: your feature"
git push origin feature/your-feature
```

### Before Deployment
```bash
# Run full evaluation
python test_evaluation.py --mode full

# Check results
# - Pass rate ≥90%
# - No performance regressions
# - All critical queries passing
```

## Known Issues

See `COMPREHENSIVE_AUDIT.md` for detailed analysis. Key issues:

### CRITICAL
1. **SQL Generation for Aggregations:** LLM may not generate correct SUM/GROUP BY for totals
2. **Chart-Answer Mismatch:** Chart and answer generated separately, can be inconsistent

### HIGH Priority
3. **Monolithic Architecture:** main.py is 3,900 lines, needs refactoring
4. **Limited Test Coverage:** Need more unit tests for calculations

See audit for fixes and timelines.

## Environment Variables

Required:
```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your_key
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key  # Fallback
APP_SECRET_KEY=your_secret      # API auth
```

Optional:
```bash
MODEL_TYPE=gemini               # or 'openai'
PORT=8000
ENVIRONMENT=production          # or 'development'
```

## Deployment

### Railway
- Auto-deploys from `main` branch
- Set environment variables in Railway Dashboard
- Check logs and metrics in Railway

### Manual Docker
```bash
docker build -t energy-chatbot .
docker run -p 8000:8000 \
  -e SUPABASE_URL=$SUPABASE_URL \
  -e SUPABASE_KEY=$SUPABASE_KEY \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e APP_SECRET_KEY=$APP_SECRET_KEY \
  energy-chatbot
```

## Monitoring

### Health Checks
```bash
# Check metrics
curl http://localhost:8000/metrics

# Run evaluation
curl -H "X-App-Key: $API_KEY" \
  "http://localhost:8000/evaluate?mode=quick"
```

### Key Metrics to Monitor
- Pass rate (target: ≥90%)
- Response time (simple: <5s, complex: <20s)
- Cache hit rate (target: ≥60%)
- Error rate (target: <5%)

## Contributing

1. Read `docs/DEVELOPER_GUIDE.md`
2. Check `COMPREHENSIVE_AUDIT.md` for priority issues
3. Create feature branch
4. Make changes with tests
5. Run `python test_evaluation.py --mode quick`
6. Create pull request

### Code Style
- PEP 8 compliant
- Type hints preferred
- Docstrings for public functions

### Commit Messages
```
feat: Add aggregation intent detection
fix: Correct share calculation
docs: Update evaluation guide
perf: Improve LLM caching
test: Add unit tests for SQL generation
```

## License

[Your License]

## Support

For questions or issues:
1. Check documentation in `docs/`
2. Review `COMPREHENSIVE_AUDIT.md` for known issues
3. Check Railway logs for errors
4. Run `/metrics` endpoint for diagnostics

## Version

**Current:** v18.7 (Gemini Analyst)

See `docs/CHANGELOG.md` for version history.
