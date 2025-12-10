# BEST PRACTICES REVIEW: LLM-Powered Text-to-SQL Bot
**Date**: 2025-11-10
**Reviewer**: Architecture Analysis vs Industry Standards
**System**: Georgia Energy Market Analysis Bot (Enai)

---

## EXECUTIVE SUMMARY

This review compares the current implementation against **industry best practices** for:
- Text-to-SQL systems (research: Spider, WikiSQL, BIRD benchmarks)
- Production LLM agents (LangChain, LlamaIndex, DSPy patterns)
- Business Intelligence chatbots (ThoughtSpot, Microsoft Power BI Q&A)
- Enterprise ML systems (MLOps, monitoring, governance)

**Overall Assessment**: 🟡 **Good Foundation, Production Gaps**

### Strengths
✅ Strong domain knowledge integration
✅ Security-conscious SQL validation
✅ Semantic-aware chart selection
✅ Multi-language support
✅ Fallback strategies

### Critical Gaps
❌ **No evaluation framework** (biggest gap for text-to-SQL)
❌ **No SQL execution validation** (cost estimation, timeouts)
❌ **Hardcoded prompts** (prevents iteration)
❌ **No caching layer** (expensive LLM calls)
❌ **Limited testing** (~9% coverage estimated)
❌ **No monitoring dashboards** (production blind spots)

---

## PART 1: TEXT-TO-SQL BEST PRACTICES

### 1.1 Query Understanding

#### ✅ What You're Doing Well
- **Multi-stage reasoning**: Domain reasoning → SQL generation → Summarization
- **Few-shot examples**: 13 diverse examples covering common patterns
- **Schema-aware**: Full database schema provided to LLM
- **Intent classification**: Detect analysis mode (light, analyst, etc.)

#### ❌ What's Missing

**1. Query Decomposition** (Research: DAIL-SQL, DIN-SQL patterns)
```python
# Missing: Complex queries should be decomposed into sub-questions
# Example: "Compare balancing price to tariffs in 2024" should become:
#   1. Get balancing prices for 2024
#   2. Get tariffs for 2024
#   3. Join and compare

# Best Practice Pattern:
def decompose_query(query: str) -> List[str]:
    """Break complex queries into simpler sub-queries."""
    # Use LLM to identify if query requires multiple steps
    # Generate sub-queries that can be executed independently
    # Compose final answer from sub-results
    pass
```

**2. Schema Linking** (Research: RAT-SQL, RYANSQL)
```python
# Missing: Explicit column/table linking before SQL generation
# Current: Relies on LLM to infer from full schema (error-prone)

# Best Practice Pattern:
def link_query_to_schema(query: str, schema: Dict) -> Dict[str, List[str]]:
    """Identify relevant tables/columns for query."""
    return {
        "tables": ["price_with_usd", "trade_derived_entities"],
        "columns": ["p_bal_gel", "entity", "date"],
        "confidence": 0.92
    }
```

**3. Query Type Classification Beyond Intent** (Missing)
```python
# Current: Only classify_query_type for chart necessity
# Missing: SQL complexity classification

# Best Practice Pattern:
SQL_COMPLEXITY = {
    "simple_select": "SELECT * FROM table WHERE...",
    "aggregate": "GROUP BY, COUNT, SUM, AVG",
    "join": "Multi-table JOIN operations",
    "subquery": "Nested SELECT statements",
    "window": "ROW_NUMBER(), RANK(), LAG()",
    "recursive": "WITH RECURSIVE CTEs"
}

def classify_sql_complexity(sql: str) -> str:
    # Use this to apply different validation/timeout strategies
    pass
```

**Recommendation**: ⭐⭐⭐ **HIGH PRIORITY**
- Add query decomposition for multi-part questions
- Implement schema linking to reduce hallucinations
- Classify SQL complexity for adaptive validation

---

### 1.2 SQL Generation

#### ✅ What You're Doing Well
- **Structured output**: LLM returns JSON with plan + SQL
- **Column aliasing**: Force English aliases for consistency
- **Synonym handling**: Auto-correct common table/column mistakes
- **Domain knowledge injection**: Conditional based on query focus

#### ❌ What's Missing

**1. SQL Validation Chain** (Missing execution safety)
```python
# Current: Only validates whitelist and sanitizes
# Missing: Execution cost/time estimation

# Best Practice Pattern from Production Text-to-SQL:
class SQLValidator:
    def validate(self, sql: str) -> ValidationResult:
        checks = [
            self.check_syntax(),           # ✅ You have this (sqlglot)
            self.check_whitelist(),        # ✅ You have this
            self.estimate_cost(),          # ❌ Missing
            self.estimate_rows(),          # ❌ Missing
            self.check_cartesian(),        # ❌ Missing (CROSS JOIN DoS)
            self.check_recursion_depth(),  # ❌ Missing
            self.validate_aggregates(),    # ❌ Missing
        ]
        return ValidationResult(passed=all(checks), warnings=[...])
```

**2. SQL Execution Sandboxing** (Critical for production)
```python
# Current: Direct database execution with MAX_ROWS limit
# Missing: Query timeout, resource limits, read-only enforcement

# Best Practice Pattern:
class SafeExecutor:
    def execute(self, sql: str, timeout_ms: int = 5000):
        # Set statement timeout
        connection.execute(text("SET statement_timeout = :timeout"),
                          {"timeout": timeout_ms})

        # Set work_mem limit (prevent memory DoS)
        connection.execute(text("SET work_mem = '128MB'"))

        # Execute in read-only transaction
        with connection.begin() as trans:
            trans.execute(text("SET TRANSACTION READ ONLY"))
            result = trans.execute(text(sql))

        return result
```

**3. SQL Self-Correction Loop** (Research: DAIL-SQL, RESDSQL)
```python
# Current: Single-shot SQL generation with retry on LLM failure
# Missing: Iterative refinement based on execution errors

# Best Practice Pattern:
MAX_ITERATIONS = 3
for attempt in range(MAX_ITERATIONS):
    sql = generate_sql(query, schema, previous_error)
    result, error = execute_safe(sql)

    if result:
        return result

    # Use error message to refine SQL
    if "column does not exist" in error:
        # Current: You do this once via synonym replacement
        # Better: Feed error back to LLM for correction
        pass
    elif "division by zero" in error:
        sql = add_nullif_guards(sql)
    else:
        break
```

**4. SQL Template Library** (Optimization opportunity)
```python
# Current: Generate all SQL from scratch
# Missing: Template matching for common queries

# Best Practice Pattern:
SQL_TEMPLATES = {
    "entity_list": "SELECT DISTINCT entity FROM {table} ORDER BY entity",
    "monthly_average": """
        SELECT DATE_TRUNC('month', date) as month, AVG({metric}) as avg_{metric}
        FROM {table} WHERE date >= :start_date GROUP BY 1 ORDER BY 1
    """,
    "year_over_year": """
        WITH current AS (SELECT ... WHERE EXTRACT(YEAR FROM date) = :year),
             previous AS (SELECT ... WHERE EXTRACT(YEAR FROM date) = :year - 1)
        SELECT c.*, p.*, (c.value - p.value) / p.value as yoy_change
        FROM current c LEFT JOIN previous p ON ...
    """
}

def try_template_match(query: str) -> Optional[str]:
    # Match simple queries to templates (faster, cheaper, more reliable)
    if re.match(r"list all (entities|types|categories)", query):
        return SQL_TEMPLATES["entity_list"].format(table="trade_derived_entities")
    return None
```

**Recommendation**: ⭐⭐⭐ **HIGH PRIORITY**
- Add SQL cost estimation (prevent expensive queries)
- Implement execution sandboxing with timeouts
- Add self-correction loop for common errors
- Build template library for frequent queries

---

### 1.3 Evaluation Framework

#### ❌ **BIGGEST GAP** - No Evaluation System

**Research Standard**: Text-to-SQL systems are evaluated on:
1. **Execution Accuracy** (EX): Does SQL return correct results?
2. **Exact Match** (EM): Does generated SQL match gold SQL?
3. **Component Match**: JOIN, WHERE, GROUP BY correctness
4. **Test Execution Accuracy** (test-suite): Pass functional tests

#### Missing: Evaluation Dataset
```python
# Best Practice: Create evaluation dataset with golden answers
EVAL_DATASET = [
    {
        "query": "What was the balancing price in June 2024?",
        "language": "en",
        "expected_sql": "SELECT AVG(p_bal_gel) FROM price_with_usd WHERE...",
        "expected_answer": "137.2 GEL/MWh",
        "expected_chart": None,  # Simple value query shouldn't chart
        "reasoning_check": "Should NOT discuss composition/xrate for simple lookup"
    },
    {
        "query": "List all entities selling on balancing market",
        "expected_sql": "SELECT DISTINCT entity FROM trade_derived_entities WHERE...",
        "expected_chart": None,  # List query shouldn't chart
        "sql_constraints": ["NO GROUP BY", "NO JOIN"],
    },
    # ... 50-100 examples covering:
    # - Simple lookups
    # - Aggregations
    # - Time series
    # - Comparisons
    # - Multi-table joins
    # - Edge cases (empty results, NULL values)
]
```

#### Missing: Automated Testing
```python
# Best Practice: Continuous evaluation on golden dataset

def run_evaluation():
    results = []
    for test_case in EVAL_DATASET:
        response = ask_endpoint(test_case["query"])

        # Check SQL correctness
        sql_correct = compare_sql(response.sql, test_case["expected_sql"])

        # Check execution accuracy
        exec_correct = compare_results(response.data, test_case["expected_answer"])

        # Check chart appropriateness
        chart_correct = (response.chart_type is None) == (test_case["expected_chart"] is None)

        # Check reasoning quality
        reasoning_passed = check_reasoning(response.answer, test_case["reasoning_check"])

        results.append({
            "query": test_case["query"],
            "sql_correct": sql_correct,
            "exec_correct": exec_correct,
            "chart_correct": chart_correct,
            "reasoning_passed": reasoning_passed,
        })

    # Report metrics
    print(f"SQL Accuracy: {sum(r['sql_correct'] for r in results) / len(results):.1%}")
    print(f"Execution Accuracy: {sum(r['exec_correct'] for r in results) / len(results):.1%}")
    print(f"Chart Accuracy: {sum(r['chart_correct'] for r in results) / len(results):.1%}")
```

**Recommendation**: ⭐⭐⭐ **HIGHEST PRIORITY**
- Create evaluation dataset with 50-100 test cases
- Implement automated evaluation pipeline
- Track metrics over time (regression detection)
- Run evaluation on every prompt/code change

---

### 1.4 Error Handling & Robustness

#### ✅ What You're Doing Well
- Retry logic with exponential backoff
- Fallback strategies (pivot injection, synonym replacement)
- Graceful degradation on LLM failures

#### ❌ What's Missing

**1. Circuit Breaker Pattern** (Prevent cascading failures)
```python
# Current: Retries blindly even if LLM is consistently failing
# Missing: Stop retrying after threshold

from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def call_llm_with_circuit_breaker(prompt):
    # After 5 failures, circuit "opens" - stop calling LLM for 60s
    # Prevents hammering failed service
    return llm.invoke(prompt)
```

**2. User Feedback Loop** (Learn from failures)
```python
# Missing: Allow users to report wrong answers

@app.post("/feedback")
def submit_feedback(
    query: str,
    response_id: str,
    feedback_type: str,  # "wrong_sql", "wrong_answer", "wrong_chart", "good"
    correction: Optional[str] = None
):
    # Store feedback for:
    # 1. Identifying common failure patterns
    # 2. Building evaluation dataset
    # 3. Fine-tuning prompts
    # 4. Creating SQL templates for frequent queries
    store_feedback(query, response_id, feedback_type, correction)
```

**3. Graceful Timeout Handling** (Known issue: 26s LLM timeout)
```python
# Current: Fixed 2-attempt retry
# Missing: Progressive timeout with early termination

import asyncio

async def llm_with_timeout(prompt: str, timeout_s: int = 10):
    try:
        result = await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=timeout_s
        )
        return result
    except asyncio.TimeoutError:
        # Return partial answer or cached response
        log.warning(f"LLM timeout after {timeout_s}s, using fallback")
        return generate_fallback_answer(prompt)
```

**Recommendation**: ⭐⭐ **MEDIUM PRIORITY**
- Implement circuit breaker for LLM calls
- Add user feedback endpoint
- Fix 26s timeout with progressive timeout strategy

---

## PART 2: LLM AGENT BEST PRACTICES

### 2.1 Prompt Engineering

#### ✅ What You're Doing Well
- Conditional domain knowledge (50-70% token reduction)
- Few-shot examples (13 diverse examples)
- Structured output format (JSON)
- Multi-language support

#### ❌ What's Missing

**1. Prompt Versioning & A/B Testing** (Critical for iteration)
```python
# Current: Prompts hardcoded in functions
# Missing: Externalized prompts with versioning

# Best Practice Pattern:
prompts/
  ├── domain_reasoning/
  │   ├── v1.yaml
  │   ├── v2.yaml (added more examples)
  │   └── v3.yaml (refined instructions)
  ├── sql_generation/
  │   ├── v1.yaml
  │   └── v2.yaml (added balancing guidance)
  └── summarization/
      └── v1.yaml

# Load prompts with version selection
PROMPT_REGISTRY = PromptRegistry("prompts/")
prompt = PROMPT_REGISTRY.get("sql_generation", version="v2")

# A/B testing
if user_id % 2 == 0:
    prompt = PROMPT_REGISTRY.get("sql_generation", version="v2")
else:
    prompt = PROMPT_REGISTRY.get("sql_generation", version="v3_experimental")
```

**2. Prompt Optimization Framework** (DSPy pattern)
```python
# Current: Manual prompt tuning
# Missing: Automated prompt optimization

# Best Practice: DSPy-style prompt optimization
class SQLGenerationModule(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("schema, query -> sql, reasoning")

    def forward(self, schema, query):
        return self.generate(schema=schema, query=query)

# Optimize prompts using evaluation dataset
optimizer = dspy.BootstrapFewShot(metric=execution_accuracy)
optimized_module = optimizer.compile(
    SQLGenerationModule(),
    trainset=EVAL_DATASET[:30],
    valset=EVAL_DATASET[30:]
)
```

**3. Prompt Token Budget** (Cost control)
```python
# Current: No token counting
# Missing: Token budget enforcement

def truncate_domain_knowledge(domain_json: str, max_tokens: int = 2000) -> str:
    # Use tiktoken to count tokens
    tokens = tiktoken.encode(domain_json)

    if len(tokens) > max_tokens:
        # Truncate to most relevant sections
        truncated = prioritize_sections(domain_json, query_focus)
        log.warning(f"Truncated domain knowledge: {len(tokens)} → {max_tokens} tokens")
        return truncated

    return domain_json
```

**Recommendation**: ⭐⭐⭐ **HIGH PRIORITY**
- Extract prompts to YAML/JSON files
- Implement prompt versioning
- Add A/B testing framework
- Add token counting and budget enforcement

---

### 2.2 LLM Provider Management

#### ✅ What You're Doing Well
- Support multiple providers (Gemini, OpenAI)
- Environment-based model selection
- Retry logic with tenacity

#### ❌ What's Missing

**1. LLM Provider Abstraction** (Reduce vendor lock-in)
```python
# Current: Direct LangChain provider calls
# Missing: Abstraction layer for easy provider switching

# Best Practice Pattern:
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_structured(self, prompt: str, schema: Dict) -> Dict:
        pass

class GeminiProvider(LLMProvider):
    def generate(self, prompt, **kwargs):
        return ChatGoogleGenerativeAI(**kwargs).invoke(prompt)

class OpenAIProvider(LLMProvider):
    def generate(self, prompt, **kwargs):
        return ChatOpenAI(**kwargs).invoke(prompt)

# Factory pattern
def get_llm_provider(model_type: str) -> LLMProvider:
    providers = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,  # Easy to add
        "local": LocalLLMProvider,        # For testing
    }
    return providers[model_type]()
```

**2. LLM Response Caching** (Reduce cost & latency)
```python
# Current: No caching - every query hits LLM
# Missing: Redis cache for identical queries

import redis
import hashlib

cache = redis.Redis(host='localhost', port=6379, db=0)

def cached_llm_call(prompt: str, ttl: int = 3600) -> str:
    # Hash prompt for cache key
    cache_key = f"llm:{hashlib.sha256(prompt.encode()).hexdigest()}"

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        log.info("✅ Cache hit for LLM call")
        return cached.decode()

    # Cache miss - call LLM
    response = llm.invoke(prompt)
    cache.setex(cache_key, ttl, response.content)

    return response.content
```

**3. Streaming Responses** (Better UX)
```python
# Current: Wait for full response before returning
# Missing: Stream tokens as they arrive

@app.post("/ask/stream")
async def ask_stream(q: Question):
    async def generate():
        # Stream domain reasoning
        async for chunk in llm.astream(domain_reasoning_prompt):
            yield f"data: {json.dumps({'type': 'reasoning', 'content': chunk})}\n\n"

        # Stream SQL generation
        async for chunk in llm.astream(sql_generation_prompt):
            yield f"data: {json.dumps({'type': 'sql', 'content': chunk})}\n\n"

        # Execute SQL and stream results
        results = execute_sql(sql)
        yield f"data: {json.dumps({'type': 'results', 'data': results})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

**Recommendation**: ⭐⭐ **MEDIUM PRIORITY**
- Add LLM provider abstraction layer
- Implement Redis caching for LLM responses
- Add streaming API endpoint for better UX

---

### 2.3 Agent Orchestration

#### ❌ Missing: Agent Framework

**Current**: Sequential LLM calls (domain reasoning → SQL → summarization)
**Missing**: Proper agent pattern with tools and decision-making

```python
# Best Practice: LangChain Agent Pattern

from langchain.agents import Tool, AgentExecutor, create_react_agent

tools = [
    Tool(
        name="GetDomainKnowledge",
        func=lambda q: get_relevant_domain_knowledge(q),
        description="Get relevant domain knowledge for a query about Georgian energy market"
    ),
    Tool(
        name="ExecuteSQL",
        func=lambda sql: execute_safe(sql),
        description="Execute SQL query on energy database. Input must be valid SELECT query."
    ),
    Tool(
        name="CalculateStatistics",
        func=lambda data: compute_stats(data),
        description="Calculate statistics (CAGR, correlations, seasonal patterns) on data"
    ),
    Tool(
        name="GenerateChart",
        func=lambda data: generate_chart(data),
        description="Generate appropriate chart visualization for data"
    ),
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt_template
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,
    verbose=True
)

# Agent decides which tools to use and in what order
response = executor.invoke({"input": user_query})
```

**Recommendation**: ⭐ **LOW PRIORITY** (current sequential approach works for now)
- Consider agent pattern if adding more complex workflows
- Current sequential approach is simpler and more debuggable

---

## PART 3: PRODUCTION OPERATIONS

### 3.1 Observability

#### ✅ What You're Doing Well
- Request ID tracking
- Basic metrics collection
- Execution timing throughout flow
- Structured logging with emojis

#### ❌ What's Missing

**1. Distributed Tracing** (Critical for debugging)
```python
# Current: Logging only
# Missing: OpenTelemetry tracing

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

tracer = trace.get_tracer(__name__)

@app.post("/ask")
async def ask(q: Question):
    with tracer.start_as_current_span("ask_endpoint") as span:
        span.set_attribute("query", q.query)
        span.set_attribute("user_id", q.user_id)

        with tracer.start_as_current_span("domain_reasoning"):
            reasoning = llm_analyze_with_domain_knowledge(q.query)
            span.set_attribute("reasoning", reasoning)

        with tracer.start_as_current_span("sql_generation"):
            plan = llm_generate_plan_and_sql(...)
            span.set_attribute("sql", plan["sql"])

        with tracer.start_as_current_span("sql_execution"):
            result = execute_sql(plan["sql"])
            span.set_attribute("row_count", len(result))

        return response

# View traces in Jaeger/DataDog
```

**2. Prometheus Metrics** (Standard monitoring)
```python
# Current: In-memory metrics (lost on restart)
# Missing: Prometheus-compatible metrics

from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
request_count = Counter('enai_requests_total', 'Total requests', ['endpoint', 'status'])
request_duration = Histogram('enai_request_duration_seconds', 'Request duration', ['endpoint'])
llm_call_duration = Histogram('enai_llm_duration_seconds', 'LLM call duration', ['model'])
active_requests = Gauge('enai_active_requests', 'Currently active requests')

# Instrument code
@app.post("/ask")
@request_duration.labels(endpoint="/ask").time()
async def ask(q: Question):
    active_requests.inc()
    try:
        response = process_query(q)
        request_count.labels(endpoint="/ask", status="success").inc()
        return response
    except Exception as e:
        request_count.labels(endpoint="/ask", status="error").inc()
        raise
    finally:
        active_requests.dec()

# Expose metrics endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

**3. Structured Logging** (JSON format)
```python
# Current: Human-readable logs with emojis
# Missing: Machine-readable JSON logs

import structlog

log = structlog.get_logger()

log.info(
    "request_started",
    request_id=request_id,
    query=user_query,
    language=detected_language,
    user_id=user_id
)

log.info(
    "sql_executed",
    request_id=request_id,
    sql=sql,
    row_count=len(results),
    duration_ms=duration * 1000
)

# Output:
# {"event": "sql_executed", "request_id": "abc123", "sql": "SELECT...", "row_count": 42, "duration_ms": 245}
```

**Recommendation**: ⭐⭐⭐ **HIGH PRIORITY**
- Add OpenTelemetry tracing (Jaeger/DataDog/Honeycomb)
- Export Prometheus metrics
- Switch to structured JSON logging
- Create monitoring dashboard (Grafana)

---

### 3.2 Performance & Scalability

#### ❌ Critical Issues

**1. Known Bottleneck: 26s LLM Timeout** (Documented but unfixed)
```python
# From PERFORMANCE_ANALYSIS.md:
# "llm_summarize took 26.32 seconds"
# Root causes: Large prompts, network latency, no streaming

# Immediate fix:
def llm_summarize_optimized(user_query, data_preview, stats_hint):
    # 1. Reduce token count
    data_preview = truncate_preview(data_preview, max_rows=10)  # Was unlimited

    # 2. Streaming response
    response = llm.stream(prompt)
    partial = ""
    for chunk in response:
        partial += chunk
        # Return early if critical info received
        if len(partial) > 500 and "\n\n" in partial:  # Complete paragraph
            return partial

    # 3. Timeout guard
    return await asyncio.wait_for(llm.ainvoke(prompt), timeout=10)
```

**2. Synchronous Architecture** (Scalability limit)
```python
# Current: Synchronous blocking calls
# Missing: Async/await for concurrent request handling

# Migration path:
async def ask_async(q: Question):
    # Domain reasoning (can run in parallel with schema fetch)
    reasoning_task = asyncio.create_task(llm_analyze_async(q.query))
    schema_task = asyncio.create_task(get_schema_async())

    reasoning, schema = await asyncio.gather(reasoning_task, schema_task)

    # SQL generation
    plan = await llm_generate_plan_async(q.query, schema, reasoning)

    # Execute SQL (blocking - can't avoid, but other requests can run)
    result = await asyncio.to_thread(execute_sql, plan["sql"])

    # Parallel summarization + chart generation
    summary_task = asyncio.create_task(llm_summarize_async(...))
    chart_task = asyncio.create_task(generate_chart_async(...))

    summary, chart = await asyncio.gather(summary_task, chart_task)

    return APIResponse(answer=summary, chart_data=chart, ...)
```

**3. No Connection Pooling Optimization** (Documented issue)
```python
# From CODE_REVIEW_REPORT.md:
# "pool_size=5, max_overflow=10 may be inadequate under load"

# Fix:
ENGINE = create_engine(
    SUPABASE_DB_URL,
    poolclass=QueuePool,
    pool_size=20,              # Up from 5
    max_overflow=30,           # Up from 10
    pool_timeout=30,
    pool_pre_ping=True,        # Verify connections before use
    pool_recycle=3600,         # Recycle connections hourly
    echo=False,
    connect_args={
        "connect_timeout": 10,
        "options": "-c statement_timeout=30000"  # 30s query timeout
    }
)
```

**4. No Caching Layer** (Expensive LLM calls repeated)
```python
# Current: Every "list all entities" query regenerates SQL and hits LLM
# Missing: Multi-tier caching

# Best Practice:
@app.post("/ask")
@cache(ttl=3600, key_func=lambda q: f"query:{hash(q.query)}")
async def ask(q: Question):
    # Layer 1: Application cache (Redis)
    # - Cache final responses for identical queries
    # - TTL: 1 hour for most queries, 5 minutes for "latest" queries

    # Layer 2: LLM response cache
    # - Cache LLM outputs for identical prompts
    # - TTL: 24 hours

    # Layer 3: SQL result cache
    # - Cache SQL execution results
    # - TTL: 5 minutes (data freshness)

    pass
```

**Recommendation**: ⭐⭐⭐ **HIGHEST PRIORITY**
- Fix 26s LLM timeout immediately
- Migrate to async/await architecture
- Optimize connection pooling
- Add Redis caching layer

---

### 3.3 Security & Governance

#### ✅ What You're Doing Well
- API key authentication
- SQL whitelist validation
- AST-based SQL parsing
- Read-only SQL enforcement
- Confidentiality rules (PPA pricing)

#### ❌ What's Missing

**1. Rate Limiting** (Prevent abuse)
```python
# Missing: No rate limiting on /ask endpoint

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/ask")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def ask(q: Question, request: Request):
    pass

# Advanced: Per-user rate limiting
@limiter.limit("100/hour", key_func=lambda: request.headers.get("X-User-ID"))
async def ask(q: Question, request: Request):
    pass
```

**2. PII Detection** (Data governance)
```python
# Missing: Detect and redact sensitive data in queries/responses

def detect_pii(text: str) -> List[str]:
    """Detect Georgian phone numbers, emails, etc."""
    patterns = {
        "phone": r"\+995\d{9}",
        "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "georgian_id": r"\d{11}",  # Georgian ID number
    }

    findings = []
    for pii_type, pattern in patterns.items():
        if re.search(pattern, text):
            findings.append(pii_type)

    if findings:
        log.warning(f"PII detected in query: {findings}")
        # Optionally: Redact or reject

    return findings
```

**3. Audit Logging** (Compliance)
```python
# Missing: Comprehensive audit trail

class AuditLog:
    def log_query(
        self,
        user_id: str,
        query: str,
        sql_generated: str,
        results_row_count: int,
        timestamp: datetime,
        ip_address: str
    ):
        # Store in separate audit database (append-only)
        # Required for:
        # - Compliance (GDPR, data access logs)
        # - Security (detect abuse patterns)
        # - Analytics (popular queries, user behavior)
        audit_db.insert({
            "user_id": user_id,
            "query": query,
            "sql": sql_generated,
            "row_count": results_row_count,
            "timestamp": timestamp,
            "ip": ip_address,
        })
```

**4. Data Access Controls** (Row-level security)
```python
# Current: All users see all data
# Missing: Row-level security for sensitive data

def inject_rls_filters(sql: str, user_id: str) -> str:
    """Add row-level security filters based on user permissions."""
    user_permissions = get_user_permissions(user_id)

    if user_permissions.role == "internal":
        # Internal users see everything
        return sql
    elif user_permissions.role == "external":
        # External users: filter to public data only
        # e.g., Add WHERE entity NOT IN ('confidential_entities')
        return add_where_clause(sql, "entity NOT IN ('renewable_ppa', 'thermal_ppa')")

    return sql
```

**Recommendation**: ⭐⭐ **MEDIUM PRIORITY**
- Add rate limiting (prevent abuse)
- Implement audit logging (compliance)
- Add PII detection (data governance)
- Consider row-level security if multi-tenant

---

## PART 4: CODE QUALITY & MAINTAINABILITY

### 4.1 Architecture

#### ❌ Issues

**1. Monolithic main.py (3,459 lines)** - Violates Single Responsibility Principle
```python
# Current structure:
main.py (3,459 lines)
├── LLM functions (3 functions, ~600 lines)
├── SQL generation & validation (10 functions, ~500 lines)
├── Data processing & statistics (15 functions, ~800 lines)
├── Chart generation (8 functions, ~600 lines)
├── API endpoints (5 endpoints, ~200 lines)
└── Utilities & config (~759 lines)

# Recommended refactoring:
src/
├── api/
│   ├── endpoints.py          # FastAPI routes
│   └── middleware.py         # Auth, CORS, request ID
├── llm/
│   ├── providers.py          # LLM provider abstraction
│   ├── prompts.py            # Prompt templates
│   └── reasoning.py          # Domain reasoning, SQL gen, summarization
├── database/
│   ├── executor.py           # Safe SQL execution
│   ├── validator.py          # SQL validation & sanitization
│   └── schema.py             # Schema management
├── analysis/
│   ├── statistics.py         # CAGR, correlations, seasonal
│   └── transformations.py   # Data processing utilities
├── visualization/
│   ├── chart_selector.py     # Dimension detection, chart type logic
│   └── chart_builder.py      # Chart data formatting
├── domain/
│   └── knowledge.py          # Domain knowledge loading
└── utils/
    ├── metrics.py            # Metrics collection
    └── logging.py            # Structured logging setup
```

**2. Hardcoded Configuration** - Should be externalized
```python
# Move to config files:
config/
├── database.yaml
│   ├── connection_string: ${SUPABASE_DB_URL}
│   └── pool_settings:
│         pool_size: 20
│         max_overflow: 30
├── llm.yaml
│   ├── default_provider: gemini
│   ├── models:
│         gemini: gemini-2.5-flash
│         openai: gpt-4o-mini
│   └── timeouts:
│         reasoning: 10
│         sql_generation: 15
│         summarization: 10
├── prompts/
│   ├── domain_reasoning_v2.yaml
│   ├── sql_generation_v3.yaml
│   └── summarization_v1.yaml
└── domain_knowledge/
    └── energy_market_v1.json
```

**Recommendation**: ⭐⭐ **MEDIUM PRIORITY**
- Refactor main.py into modules (aim for <500 lines per file)
- Extract configuration to YAML/JSON
- Use dependency injection for testability

---

### 4.2 Testing

#### ❌ Current Coverage: ~9% (estimated)

**Test Pyramid for LLM Apps:**
```
         /\
        /  \   E2E Tests (5-10% of tests)
       /    \  - Full user flows with real LLM
      /------\ Integration Tests (20-30%)
     /        \ - LLM + DB + API (mocked LLM)
    /          \ Unit Tests (60-75%)
   /____________\ - Pure functions, validation, transformations

Current: Only unit tests for 5-6 functions
Missing: Integration tests, E2E tests, LLM evaluation tests
```

**Recommended Test Suite:**
```python
tests/
├── unit/
│   ├── test_sql_validation.py       # Whitelist, sanitization
│   ├── test_query_classification.py # Intent, type, focus detection
│   ├── test_chart_selection.py      # Dimension inference, type selection
│   ├── test_statistics.py           # CAGR, correlations
│   └── test_transformations.py      # Data processing
├── integration/
│   ├── test_llm_flows.py            # End-to-end LLM chains (mocked)
│   ├── test_sql_execution.py        # Database queries (test DB)
│   └── test_api_endpoints.py        # FastAPI routes
├── e2e/
│   ├── test_user_scenarios.py       # Real user flows
│   └── test_evaluation_dataset.py   # Golden answer tests
└── fixtures/
    ├── sample_queries.json
    ├── sample_sql.sql
    └── sample_responses.json
```

**Recommendation**: ⭐⭐⭐ **HIGH PRIORITY**
- Increase unit test coverage to 70%+
- Add integration tests with test database
- Create E2E test suite with evaluation dataset
- Run tests in CI/CD pipeline

---

### 4.3 Documentation

#### ❌ Missing Critical Documentation

**1. README.md** (Onboarding new developers)
```markdown
# Enai - Georgian Energy Market Analysis Bot

## Overview
LLM-powered text-to-SQL bot for Georgian energy market data analysis.
Supports Georgian, Russian, and English queries.

## Features
- Natural language to SQL translation
- Domain-aware query understanding
- Automatic chart generation
- Multi-language support

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run tests
pytest

# Start server
python main.py
```

## Architecture
[Include diagram]

## API Documentation
[Link to OpenAPI docs]

## Development Guide
[How to add features, test changes]
```

**2. API Documentation** (OpenAPI/Swagger)
```python
# Add docstrings to endpoints
@app.post(
    "/ask",
    response_model=APIResponse,
    summary="Ask a question about Georgian energy market",
    description="""
    Accepts natural language questions in Georgian, Russian, or English.
    Returns answer text, chart data, and metadata.

    Example queries:
    - "What was the balancing price in June 2024?"
    - "Show me generation by technology"
    - "Compare tariffs across entities"
    """,
    responses={
        200: {"description": "Successful response with answer and chart"},
        400: {"description": "Invalid query or unsafe SQL"},
        401: {"description": "Unauthorized - missing or invalid API key"},
        500: {"description": "Server error - LLM failure or database issue"}
    }
)
async def ask(q: Question):
    pass
```

**Recommendation**: ⭐⭐ **MEDIUM PRIORITY**
- Create comprehensive README.md
- Add OpenAPI documentation to endpoints
- Create architecture diagrams
- Write development guide

---

## PART 5: PRIORITIZED ACTION PLAN

### 🔴 **P0 - CRITICAL (Do First)**

1. **Fix 26s LLM Timeout**
   - Impact: Blocking user experience issue
   - Effort: 1 day
   - Solution: Streaming, prompt reduction, timeout guards

2. **Create Evaluation Dataset**
   - Impact: Enables quality measurement and regression detection
   - Effort: 3-5 days (create 50-100 test cases)
   - Solution: Build golden answer dataset, automated evaluation

3. **Add SQL Execution Safeguards**
   - Impact: Prevent DoS, expensive queries
   - Effort: 2 days
   - Solution: Query timeout, cost estimation, connection pooling

4. **Implement Caching Layer**
   - Impact: 50%+ cost reduction, 3-5x latency improvement
   - Effort: 2-3 days
   - Solution: Redis cache for LLM responses and SQL results

### 🟡 **P1 - HIGH (Do Soon)**

5. **Extract Prompts to Config Files**
   - Impact: Enable rapid iteration, A/B testing
   - Effort: 2 days
   - Solution: YAML prompt templates, versioning

6. **Add OpenTelemetry Tracing**
   - Impact: Essential for debugging production issues
   - Effort: 1-2 days
   - Solution: Jaeger/DataDog integration

7. **Increase Test Coverage to 70%+**
   - Impact: Catch regressions, safer refactoring
   - Effort: 5-7 days
   - Solution: Unit tests for all functions, integration tests

8. **Schema Linking & Query Decomposition**
   - Impact: Reduce SQL hallucinations by 30-40%
   - Effort: 3-4 days
   - Solution: Identify relevant tables/columns before SQL gen

### 🟢 **P2 - MEDIUM (Do Eventually)**

9. **Refactor main.py into Modules**
   - Impact: Better maintainability
   - Effort: 5-7 days
   - Solution: Split into api/, llm/, database/, analysis/, viz/ modules

10. **Migrate to Async Architecture**
    - Impact: 3-5x concurrency improvement
    - Effort: 7-10 days
    - Solution: async/await, parallel LLM calls

11. **Add Rate Limiting & Audit Logging**
    - Impact: Security, compliance
    - Effort: 2-3 days
    - Solution: slowapi rate limiting, audit database

12. **Create Documentation**
    - Impact: Easier onboarding
    - Effort: 3-4 days
    - Solution: README, API docs, architecture diagrams

---

## SUMMARY SCORECARD

| Category | Score | Grade |
|----------|-------|-------|
| **Text-to-SQL Quality** | 6/10 | 🟡 B- |
| ↳ SQL Generation | 7/10 | Good prompt engineering, few-shot examples |
| ↳ Evaluation | 2/10 | ❌ No golden dataset, no automated testing |
| ↳ Error Correction | 6/10 | Some fallbacks, but no self-correction loop |
| **LLM Agent Design** | 6.5/10 | 🟡 B |
| ↳ Prompt Engineering | 8/10 | Conditional prompts, structured output |
| ↳ Configuration | 4/10 | ❌ Hardcoded prompts, no A/B testing |
| ↳ Provider Management | 6/10 | Multi-provider, but no caching |
| **Production Readiness** | 4/10 | 🔴 C- |
| ↳ Performance | 3/10 | ❌ 26s timeout, no async, no caching |
| ↳ Observability | 5/10 | Basic logging, needs tracing & metrics |
| ↳ Security | 7/10 | Good SQL validation, missing rate limiting |
| **Code Quality** | 5/10 | 🟡 C+ |
| ↳ Testing | 2/10 | ❌ ~9% coverage, no integration tests |
| ↳ Architecture | 4/10 | Monolithic design, needs refactoring |
| ↳ Documentation | 3/10 | ❌ No README, no API docs |

**Overall Assessment**: 5.4/10 - 🟡 **Good Foundation, Production Gaps**

---

## KEY TAKEAWAYS

### ✅ **What Makes This Good**
- Strong domain knowledge integration
- Security-conscious SQL validation
- Intelligent query classification and routing
- Semantic-aware visualization

### ❌ **What Holds This Back**
1. **No evaluation framework** - Can't measure quality or track regressions
2. **Performance bottlenecks** - 26s timeout, no caching, no async
3. **Limited testing** - High risk of breaking changes
4. **Configuration rigidity** - Hard to iterate on prompts

### 🎯 **Path to Production Excellence**

**Month 1: Quality & Performance**
- Create evaluation dataset (50-100 queries)
- Fix LLM timeout issue
- Add Redis caching
- SQL execution safeguards

**Month 2: Observability & Testing**
- OpenTelemetry tracing
- Prometheus metrics
- Increase test coverage to 70%
- Grafana dashboards

**Month 3: Architecture & Scale**
- Extract prompts to config
- Refactor into modules
- Migrate to async/await
- Connection pool optimization

**Result**: Production-ready system with 95%+ reliability, <2s p95 latency, measurable quality

---

**Review Complete**: 2025-11-10
**Confidence**: High (based on industry research and production patterns)
**Next Step**: Prioritize P0 critical items, starting with evaluation dataset
