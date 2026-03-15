# Query Pipeline Architecture

Comprehensive technical reference for the Enai query processing pipeline. Covers every stage, decision point, threshold, and fallback path from HTTP request to JSON response.

**Last updated:** 2026-03-15

---

## 1. System Overview

Enai is a conversational Q&A system for the Georgian electricity market. Users ask questions about balancing prices, generation mix, tariffs, and market composition. The system retrieves data from a PostgreSQL database, runs deterministic analysis, and returns grounded natural-language answers with optional charts.

### Key Design Principles

1. **Deterministic over generative.** Typed tools with parameterized SQL are preferred over LLM-generated SQL. The LLM is never trusted to produce numbers — all numeric claims are verified against raw database cells.
2. **Provenance-gated output.** Every numeric token in the final answer must trace back to a specific database cell with a SHA256 fingerprint. If coverage drops below 100%, the answer is replaced with a conservative fallback.
3. **Defense in depth.** A firewall blocks prompt injection before any LLM call. SQL is validated against a table whitelist. Read-only transactions are enforced. The provenance gate catches any remaining hallucination.
4. **Fast path first.** The keyword router handles >80% of queries in <500ms with zero LLM cost. The LLM analyzer and agent loop only activate when the fast path misses.

### Architecture Diagram

```
 Browser → Supabase Edge Function → Railway Backend
                                      │
                                      ├─ Stage 0:   Firewall + Prepare Context
                                      ├─ Stage 0.2: LLM Question Analyzer (optional)
                                      ├─ Stage 0.5: Keyword Router (fast path)
                                      │   ├─ HIT → Stage 0.6: Tool Execute
                                      │   │         └─ Composition Enrichment (for why-queries)
                                      │   └─ MISS → Stage 0.7: LLM Analyzer Routing (fallback)
                                      │              ├─ HIT → Tool Execute + Enrichment
                                      │              └─ MISS ─┐
                                      ├─ Agent Loop ←─────────┘ (bounded ReACT, max 3 rounds)
                                      │   ├─ data_exit → continue
                                      │   ├─ conceptual_exit → return
                                      │   └─ fallback_exit ─┐
                                      ├─ SQL Fallback ←─────┘ (LLM plan + validated SQL)
                                      ├─ Stage 3:   Analysis Enrichment (stats, correlations, trends)
                                      ├─ Stage 4:   Summarization + Provenance Gate
                                      └─ Stage 5:   Chart Pipeline
```

---

## 2. Request Lifecycle

### 2.1 Frontend → Edge Function → Railway

1. **User submits query** via the chat UI (`ChatPage.jsx`).
2. **Supabase Edge Function** (`chat-with-enerbot`) handles the request:
   - Authenticates the user via Supabase JWT token.
   - Validates `X-App-Key` header against the shared secret.
   - Loads the last 3 Q&A pairs from the `chat_history` table (6 rows, newest first, reversed to chronological order). Uses sequential role scanning to pair user→assistant messages resiliently.
   - Sends to Railway: `{ query, service_tier, conversation_history? }` with headers `X-App-Key`, `X-Request-Id`.
3. **Railway backend** (`main.py /ask POST`) receives the request:
   - Validates `X-App-Key` against `GATEWAY_SHARED_SECRET`.
   - Validates `Referer` header against allowed origins.
   - Applies **backpressure gate**: if concurrent requests exceed `ASK_MAX_CONCURRENT_REQUESTS` (default 8), returns 503.
   - Generates or validates `X-Session-Token` (HMAC-SHA256 signed).

### 2.2 Session Management

- **Session token issuance:** Railway generates an HMAC-SHA256 signed session token on first request. The token is returned in the `X-Session-Token` response header.
- **In-process session store:** Thread-safe dictionary keyed by session ID. Stores up to `SESSION_HISTORY_MAX_TURNS` (default 3) Q&A pairs per session. Sessions expire after `SESSION_IDLE_TTL_SECONDS` (default 3600).
- **History seeding:** If the in-process session has no history (e.g., first request after deploy) but the edge function provided `conversation_history`, Railway seeds the session store with those turns. Individual items are truncated to 2000 characters to prevent memory bloat.
- **History consumption:** The bound history is passed to `process_query()` and consumed by the LLM Question Analyzer (Stage 0.2) and the Summarizer (Stage 4).

---

## 3. Pipeline Stages

All stages are orchestrated by `pipeline.process_query()`. Each stage records its duration in `ctx.stage_timings_ms` and emits a structured `TRACE` log line with timing and decision data.

### 3.0 Stage 0: Prepare Context

**Function:** `planner.prepare_context(ctx)`

Three cheap heuristic checks before any LLM call:

1. **Language detection** — Identifies EN, KA (Georgian), RU, or ZH from the query text. Sets `ctx.lang_instruction` which is prepended to all subsequent LLM prompts (e.g., "Respond in Georgian").

2. **Analysis mode selection** — Classifies the query as `light` or `analyst`:
   - **Light mode** (higher priority): Triggered by simple-fact patterns — "what is", "what was", "list", "show", "give me", plus Georgian/Russian equivalents.
   - **Analyst mode**: Triggered by deep-analysis keywords — "trend over time", "correlation", "driver", "impact on", "analyze", "what drives", "what causes", or any word in the analytical keyword set (trend, change, growth, compare, volatility, etc.).
   - **Default:** `light`.

3. **Conceptual question detection** — Regex scan for definition-seeking patterns ("What is a PPA?", "რა არის...", "что такое..."). If detected, the pipeline **short-circuits** directly to `summarizer.answer_conceptual()`, bypassing all data stages. This avoids unnecessary LLM calls, tool routing, and SQL generation for pure knowledge questions.

### 3.0.2 Stage 0.2: LLM Question Analyzer

**Function:** `planner.analyze_question_active(ctx)` or `planner.analyze_question_shadow(ctx)`
**Controlled by:** `ENABLE_QUESTION_ANALYZER_HINTS` (active mode) or `ENABLE_QUESTION_ANALYZER_SHADOW` (shadow/logging-only mode)

Calls `llm_analyze_question()` which sends the query + conversation history + domain catalogs to the LLM (using `ROUTER_MODEL` if configured, otherwise the global default). Returns a structured `QuestionAnalysis` Pydantic model:

- **classification**: `query_type` (data_retrieval, data_explanation, conceptual_definition, etc.), `analysis_mode` (light/analyst), `confidence` (0.0-1.0)
- **routing**: `preferred_path` (tool/sql/knowledge/clarify), `prefer_tool` (bool)
- **tooling**: `candidate_tools` — list of `(name, score, params_hint, reason)` sorted by score
- **sql_hints**: `period` (start_date, end_date), `entities`, `metrics`
- **canonical_query_en**: English translation of the query for consistent parameter extraction

**Disagreement logging:** The trace logs whether the analyzer and heuristic disagree on conceptual classification (`conceptual_disagree`) and analysis mode (`mode_disagree`). This is essential for tuning the heuristic layer.

**Shadow mode:** When `ENABLE_QUESTION_ANALYZER_SHADOW=true` and `ENABLE_QUESTION_ANALYZER_HINTS=false`, the analyzer runs and logs results but does NOT influence routing decisions. This allows comparison logging before cutover.

### 3.0.5 Stage 0.5: Keyword Router (Fast Path)

**Function:** `router.match_tool(query, is_explanation)`
**Returns:** `ToolInvocation(name, params, confidence, reason)` or `None`

The keyword router is the primary routing mechanism — zero LLM cost, <10ms latency. It evaluates the query against four deterministic keyword chains:

**Keyword chains:**

| Tool | Trigger keywords (subset) | Example query |
|------|--------------------------|---------------|
| `get_tariffs` | tariff, regulated, gnerc, enguri, gardabani, capacity fee | "What are the regulated tariffs?" |
| `get_balancing_composition` | share, composition, mix, proportion, ppa, import share | "What is the import share?" |
| `get_generation_mix` | generation, technology, hydro, thermal, wind, solar, demand | "Show generation by technology" |
| `get_prices` | price, cost, balancing price, deregulated, exchange rate, xrate | "Balancing price trend in 2024" |

**Semantic fallback** — When no keyword chain matches, the router computes a semantic score for each tool:
- Multi-word terms (e.g., "balancing price"): +2.0 for exact match in query
- Single-word terms (e.g., "price"): +1.0 for word-level match
- Score = `min(1.0, raw_hits / max(2.0, len(terms) * 0.35))`
- **Minimum score threshold:** `ROUTER_SEMANTIC_MIN_SCORE` (default 0.62)
- **Competitive margin:** Top score must exceed second-best by at least **0.08** to avoid ambiguous routing
- **Confidence:** `min(0.95, max(0.65, score))` — bounded between 0.65 and 0.95

**is_explanation detection** — Before calling the router, the pipeline checks if the query is an explanation request. Sources (in priority order):
1. If the LLM Question Analyzer classified the query as `data_explanation` or `conceptual_definition` → `is_exp=True`
2. Otherwise, keyword scan for "why", "explain", "reason" (plus Georgian/Russian equivalents) → `is_exp=True`

This flag affects date range extraction: for "why" queries, the router automatically expands the start date to include the previous month for month-over-month delta calculations.

**Parameter extraction** — When a tool matches, the router extracts parameters using deterministic regex:
- **Dates:** "from 2020 to 2024", "2020-2024", "june 2023", standalone "2024"
- **Currency:** "in USD", "in GEL" → defaults to "GEL"
- **Price metric:** "balancing", "deregulated", "guaranteed capacity", "exchange rate" → defaults to "balancing"
- **Entities:** Tool-specific entity lists for tariffs, balancing composition, generation types

### 3.0.6 Stage 0.6: Tool Execution

**Function:** `execute_tool(invocation)` → `(DataFrame, columns, rows)`

When the router returns a match:
1. Execute the typed tool (parameterized SQL via `run_text_query()` — no LLM-generated SQL).
2. **Stamp provenance** — Record the exact columns, rows, source="tool", and a hash of the tool invocation for later verification.
3. **Relevance validation** — `validate_tool_relevance(query, tool_name)` checks whether the tool output is semantically relevant to the query. If irrelevant (e.g., router matched `get_prices` but query is "What is GNERC?"):
   - Clear the tool result (empty DataFrame)
   - Clear provenance
   - Set `ctx.used_tool = False`
   - Fall through to the next routing layer

### 3.0.7 Stage 0.7: LLM Analyzer Routing (Fallback)

**When:** The keyword router returned `None` (no match) AND `ENABLE_QUESTION_ANALYZER_HINTS=true` AND the Question Analyzer produced a result in Stage 0.2.

**Function:** `planner.build_tool_invocation_from_analysis(qa, raw_query)`

This stage converts the LLM analyzer's structured output into a concrete `ToolInvocation`. The analyzer sees conversation history, so it can resolve follow-up queries like "What about 2024?" that the keyword router cannot.

**Routing gate:**
- Primary: `preferred_path` must be `TOOL`
- Soft boost: `prefer_tool=True` allows routing for ambiguous paths (e.g., `CLARIFY`), but NOT when `preferred_path` is explicitly `SQL` or `KNOWLEDGE`
- **Minimum tool score:** 0.55 on the top candidate

**Parameter resolution cascade** (3-tier, per parameter):
1. Analyzer's `params_hint` (from the LLM)
2. Analyzer's `sql_hints.period` (structured period extraction)
3. Deterministic regex extraction (same functions as the keyword router)

If either start_date or end_date is still missing after tiers 1-2, the regex fallback fills in whichever date is absent.

**After tool execution:** Same flow as Stage 0.6 — provenance stamping, relevance validation, and fallthrough on failure.

### 3.1 Composition Enrichment (Why-Queries)

**Function:** `_enrich_prices_with_composition(ctx, invocation, is_explanation)`

**When:** `is_explanation=True` AND tool is `get_prices` AND the result has no `share_*` columns yet.

**What it does:** Automatically fetches `get_balancing_composition` with the same date range and merges the share columns (share_import, share_deregulated_hydro, share_regulated_hpp, etc.) into the price DataFrame via a left join on the date column.

**Why:** Users asking "Why did balancing prices increase?" need both the price trend AND the supply composition shift to form a valid causal answer. Without composition data, the summarizer can only describe the price change, not explain it.

**Runs identically** from both Stage 0.5 (keyword router) and Stage 0.7 (analyzer router) paths. Non-fatal — if the composition fetch fails, the pipeline continues with price data only.

### 3.2 Agent Loop

**Function:** `orchestrator.run_agent_loop(ctx)`
**When:** No tool route was used AND `ENABLE_AGENT_LOOP=true`
**Controlled by:** `AGENT_MAX_ROUNDS` (default 3)

A bounded ReACT-style agent loop that gives the LLM access to typed tools:

**Tool whitelist:** `get_prices`, `get_balancing_composition`, `get_tariffs`, `get_generation_mix`

**Per-round flow:**
1. LLM receives: system prompt (instruction + language) + user query + previous tool results
2. LLM decides: call a tool, or return a text answer
3. If tool call: execute with timeout (`AGENT_TOOL_TIMEOUT_SECONDS`, default 15s), format preview (`AGENT_TOOL_PREVIEW_ROWS`, default 10 rows; `AGENT_TOOL_PREVIEW_MAX_CHARS`, default 3000 chars)
4. Feed result back to LLM for next round

**Three exit conditions:**

| Exit | Trigger | What happens |
|------|---------|-------------|
| `data_exit` | LLM returns content + datasets collected + query prefers data | Primary dataset selected by keyword relevance scoring; attached to context for enrichment + summarization |
| `conceptual_exit` | LLM returns content + no datasets (or query doesn't prefer data) | Answer used directly; pipeline returns |
| `fallback_exit` | Model error, ambiguous datasets, empty response, max rounds exceeded | Falls through to SQL fallback |

After `data_exit`, the result is validated by `validate_tool_relevance()`. If blocked, the exit is downgraded to `fallback_exit`.

### 3.3 SQL Fallback (Legacy Path)

**When:** `ctx.used_tool` is still False after all routing attempts.

**Stage 1: Generate Plan** — `planner.generate_plan(ctx)`
- LLM generates a JSON plan + raw SQL in a single call, separated by `---SQL---`
- Plan structure: `{intent, target, period, chart_strategy?, chart_groups?}`
- Uses `PLANNER_MODEL` if configured
- Retries once on parsing failure; salvages SQL even if plan JSON is malformed
- Checks `should_skip_sql_execution()` — certain intents (pure conceptual, ambiguous) bypass SQL

**Stage 2: SQL Execution** — `sql_executor.validate_and_execute(ctx)`

SQL goes through multiple validation/repair steps:

1. **Sanitization:** Strip markdown fences, normalize whitespace
2. **Table whitelist check:** Extract table names from SQL; reject if any table not in `ALLOWED_TABLES`:
   - `dates_mv`, `entities_mv`, `price_with_usd`, `tariff_with_usd`, `tech_quantity_view`, `trade_derived_entities`, `monthly_cpi_mv`, `energy_balance_long_mv`
3. **Synonym repair:** Auto-replace common misnames:
   - Tables: "prices" → "price_with_usd", "tariffs" → "tariff_with_usd"
   - Columns: "tech_type" → "type_tech", "quantity_mwh" → "quantity_tech"
4. **LIMIT enforcement:** Ensures every query has a LIMIT clause (max `MAX_ROWS`, default 5000)
5. **Balancing pivot injection:** For share/composition queries on `trade_derived_entities`, auto-injects a CTE with CASE WHEN pivots for all entity types
6. **Tech type filtering:** Applies supply/demand/transit type filters based on query keywords

**Error recovery:**
- `UndefinedColumn` on `trade_derived_entities` → auto-inject balancing pivot CTE and retry
- `UndefinedColumn` on other tables → column synonym replacement and retry
- All other errors → propagate up

**Security:** Read-only transactions enforced. Dangerous SQL (DROP, TRUNCATE, ALTER, INSERT, DELETE, UPDATE) is blocked at the firewall stage before any SQL execution.

### 3.4 Stage 3: Analysis Enrichment

**Function:** `analyzer.enrich(ctx)`

All computation is deterministic Python — zero LLM involvement. The enrichment adds derived metrics to the context for the summarizer:

1. **Statistical analysis:**
   - Pearson correlation between price and composition share columns
   - Trend calculation (linear regression slope)
   - CAGR (compound annual growth rate) for multi-year data
   - Year-over-year comparison
   - Seasonal pattern detection (summer vs winter months)

2. **Share shift filtering:**
   - Only month-over-month balancing share shifts with absolute delta >= **0.005** (0.5 percentage points) are considered significant
   - Below this threshold, shifts are treated as noise

3. **Why-analysis signal detection:**
   - Computes `price_direction` vs `mix_pressure` (did expensive sources increase?) vs `xrate_direction`
   - **Contradiction guard:** If price direction contradicts both mix pressure and exchange rate direction, the analyzer generates a deterministic override string (e.g., "The observed composition shift points in the opposite direction..."). This string **replaces** the LLM summary to prevent hallucinated causal claims.
   - **Missing data guard:** If no share data is available, explicitly outputs "No balancing composition data was available" rather than a false "No shift observed."

4. **Trendline pre-calculation:** If enabled, computes linear trendlines with optional forward extension for chart overlays.

### 3.5 Stage 4: Summarization + Provenance Gate

**Function:** `summarizer.summarize_data(ctx)`

#### Summarization

- **Deterministic override path:** If `share_summary_override` or `why_summary_override` is populated (e.g., by the contradiction guard), the LLM path is completely bypassed. The override text becomes the answer.
- **LLM summarization path:** The LLM receives: user query + data preview (first 200 rows) + stats hint + domain knowledge. Uses `SUMMARIZER_MODEL` if configured.
- **Structured output:** `SummaryEnvelope(answer, claims, citations, confidence)`
  - `answer`: Natural-language narrative
  - `claims`: List of factual claims extracted from the answer
  - `citations`: List of data references
  - `confidence`: Float 0.0-1.0
- **Grounding check:** After generation, the system compares every numeric token in the answer+claims to the source data. Match ratio must be >= **0.7** (70%). If below, retries with `strict_grounding=True`.

#### Provenance Gate

The provenance gate is the final quality barrier. It verifies that every numeric claim in the LLM's answer traces back to an exact database cell.

**Step-by-step:**
1. Extract all numeric tokens from `ctx.summary_claim_provenance`
2. If no claims at all → gate passes (reason: `no_claims`)
3. If no numeric claims → gate passes (reason: `no_numeric_claims`)
4. For each numeric claim, check if it maps to an exact cell reference with: `(row_index, column, value, SHA256_fingerprint)`
5. Compute coverage = grounded_numeric_claims / total_numeric_claims

**Gate decision:**
- **Pass:** No ungrounded claims AND coverage >= `PROVENANCE_MIN_COVERAGE` (default **1.0** = 100%)
- **Fail:** ANY ungrounded claim OR coverage below threshold

**On failure:**
- Answer is **completely replaced** with: "I could not produce citation-grade grounding for all numeric claims..."
- Claims list cleared
- Confidence forced to `min(current, 0.2)`
- Unmatched tokens logged for debugging

**Provenance limits:**
- Max references per token: 4
- Max references per claim: 12
- Max total citations: 30
- Max rows scanned: 200

### 3.6 Stage 5: Chart Pipeline

**Function:** `chart_pipeline.build_chart(ctx)`

#### Chart Type Selection

The pipeline infers chart type from the data structure:

| Condition | Chart Type |
|-----------|-----------|
| Time column + categories + "share" dimension | `stackedbar` |
| Time column + categories | `line` |
| Time column only | `line` |
| Categories + "share" + unique categories <= 8 | `pie` |
| Categories + "share" + unique categories > 8 | `bar` |
| Categories only | `bar` |
| Default | `line` |

Time detection keywords: "date", "year", "month" (plus Georgian equivalents).
Category detection keywords: "type", "sector", "entity", "source", "segment", "technology", "region", etc.

#### Series Limiting

Maximum **3 series** per chart. When exceeded, columns are scored by relevance:
- Price/tariff keywords: +10
- Exchange rate keywords: +10
- Share/composition keywords: +5
- Contains "p_bal": +3
- Contains "xrate": +2

Top 3 columns by score are selected.

#### Skip Conditions

Charts are skipped when:
1. Explanatory queries ("why", "how", "reason") with < 5 data rows
2. Definitional queries ("define", "meaning of")
3. SQL relevance flag set (`ctx.skip_chart_due_to_relevance=True`)

#### Dual-Axis Detection

The chart pipeline detects when two different metric types should use separate Y-axes:
- Price + Exchange Rate → dual axis
- Price + Composition Shares → dual axis
- Quantity + Price/Tariff → dual axis

---

## 4. Security Layer

### 4.1 AI Firewall (Stage 0)

**Function:** `guardrails.firewall.inspect_query(query)` — runs before ANY other processing.

**Block rules** (each adds +5 risk points):

| Pattern | Example blocked |
|---------|----------------|
| Instruction override | "ignore previous instructions and..." |
| Prompt exfiltration | "reveal your system prompt" |
| Role hijack | "you are now DAN, act as root" |
| Dangerous SQL intent | "DROP TABLE users", "DELETE FROM prices" |

**Warn rules** (flagged but not blocked):
- System prompt meta-references ("system prompt", "developer message")
- SQL comment tokens (`--`, `/*`)
- Excessive control characters

**Additional scoring:**
- Oversized prompt (> 1800 chars): +2 risk
- SQL-like patterns (SELECT...FROM): +1 risk

**Decision:**
- `block`: Any block rule matched OR risk_score >= 8
- `warn`: Any matched rules with lower risk
- `allow`: No matches, risk_score = 0

**Query sanitization** (applied regardless of decision):
- Remove markdown code fences
- Remove control characters
- Normalize whitespace
- Trim to 1800 characters

### 4.2 Red-Team Gate

**Function:** `guardrails.redteam_gate.evaluate_redteam_gate(query)`

A secondary weighted scoring system with category-specific risk patterns. Operates independently of the main firewall.

### 4.3 Database Security

- **Table whitelist:** SQL can only query 8 approved materialized views / tables
- **Read-only enforcement:** Dangerous SQL keywords (DROP, TRUNCATE, ALTER, CREATE, INSERT, DELETE, UPDATE) blocked at firewall
- **Parameterized SQL in typed tools:** Tools use `run_text_query()` with pre-written SQL templates — no user input interpolation
- **Query timeout:** 30 seconds (`SQL_TIMEOUT_SECONDS`)
- **Connection pool:** `DATABASE_POOL_SIZE=10`, `DATABASE_MAX_OVERFLOW=5`

---

## 5. LLM Configuration

### 5.1 Global Model Config

| Env Var | Default | Description |
|---------|---------|-------------|
| `MODEL_TYPE` | `gemini` | Provider selection: `gemini` or `openai` |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Default Gemini model for all stages |
| `OPENAI_MODEL` | `gpt-4o-mini` | Default OpenAI model (when MODEL_TYPE=openai) |

### 5.2 Per-Stage Overrides

Each pipeline stage can use a different model. When set, overrides the global default for that stage only. Only Gemini model names are supported for overrides.

| Env Var | Stage | Typical use |
|---------|-------|-------------|
| `ROUTER_MODEL` | Question Analyzer (Stage 0.2) | Cheap/fast model for intent classification |
| `PLANNER_MODEL` | Plan + SQL Generation (Stage 1) | Capable model for SQL generation |
| `SUMMARIZER_MODEL` | Summarization (Stage 4) | Capable model for grounded narrative |

### 5.3 Model Instance Management

`get_llm_for_stage(stage_model)` in `core/llm.py`:
- If no override (or override matches global default) → returns the global singleton (zero overhead)
- If override set + `GOOGLE_API_KEY` present → creates and caches a dedicated `ChatGoogleGenerativeAI` instance
- If override set + `GOOGLE_API_KEY` missing → logs warning, falls back to global default
- Instances cached per model name for the lifetime of the process

### 5.4 Circuit Breakers

Separate circuit breakers for LLM and database:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_CB_FAILURE_THRESHOLD` | 5 | Failures before circuit opens |
| `LLM_CB_RESET_TIMEOUT_SECONDS` | 30 | Seconds before retry after open |
| `DB_CB_FAILURE_THRESHOLD` | 5 | Same, for database |
| `DB_CB_RESET_TIMEOUT_SECONDS` | 30 | Same, for database |

When a circuit breaker opens, all requests to that provider fail fast with a `RuntimeError` until the reset timeout.

### 5.5 LLM Response Cache

In-memory hash-based cache (`LLMResponseCache`):
- Key: SHA256 of the prompt string (first 16 chars)
- Max size: `CACHE_MAX_SIZE` (default 1000)
- Eviction: Oldest 10% when full (`CACHE_EVICTION_PERCENT=0.1`)
- Cached: Plan+SQL generation, domain reasoning, question analysis
- Not cached: Summarization (varies by data context)

---

## 6. Feature Flags & Configuration

### Feature Flags

| Flag | Default | Effect |
|------|---------|--------|
| `ENABLE_TYPED_TOOLS` | `true` | Enable/disable the entire typed tool routing layer (Stages 0.5-0.7) |
| `ENABLE_AGENT_LOOP` | `true` | Enable/disable the ReACT agent loop fallback |
| `ENABLE_QUESTION_ANALYZER_HINTS` | `true` | Active mode: analyzer output influences routing (Stage 0.7) |
| `ENABLE_QUESTION_ANALYZER_SHADOW` | `false` | Shadow mode: analyzer runs but doesn't influence routing (logging only) |
| `ENABLE_TRACE_DEBUG_ARTIFACTS` | `false` | Emit full debug artifacts (raw SQL, full analysis objects) in traces |

### Key Thresholds

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_ROWS` | 5000 | Maximum rows returned from any SQL query |
| `ROUTER_SEMANTIC_MIN_SCORE` | 0.62 | Minimum semantic score for router fallback match |
| `PROVENANCE_MIN_COVERAGE` | 1.0 | Minimum provenance coverage (1.0 = 100%) |
| `AGENT_MAX_ROUNDS` | 3 | Maximum agent loop iterations |
| `AGENT_TOOL_TIMEOUT_SECONDS` | 15 | Per-tool execution timeout in agent loop |
| `AGENT_TOOL_PREVIEW_ROWS` | 10 | Rows shown to LLM in tool preview |
| `AGENT_TOOL_PREVIEW_MAX_CHARS` | 3000 | Max chars in tool preview |
| `PROMPT_BUDGET_MAX_CHARS` | 30000 | Maximum prompt size sent to LLM |
| `SESSION_HISTORY_MAX_TURNS` | 3 | Q&A pairs stored per session |
| `SESSION_IDLE_TTL_SECONDS` | 3600 | Session expiry (1 hour) |
| `TRACE_TEXT_MAX_CHARS` | 800 | Max text length in trace events |
| `TRACE_MAX_LIST_ITEMS` | 8 | Max list items in trace events |

### Cost Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_INPUT_COST_PER_1K_USD` | 0 | USD per 1K input tokens (OpenAI) |
| `OPENAI_OUTPUT_COST_PER_1K_USD` | 0 | USD per 1K output tokens (OpenAI) |
| `GEMINI_INPUT_COST_PER_1K_USD` | 0 | USD per 1K input tokens (Gemini) |
| `GEMINI_OUTPUT_COST_PER_1K_USD` | 0 | USD per 1K output tokens (Gemini) |

---

## 7. Observability

### Per-Request Telemetry

Every request is assigned a `trace_id` (UUID). Per-request LLM usage is tracked via `ContextVar`:
- `llm_calls`: Number of LLM API calls in this request
- `prompt_tokens`, `completion_tokens`, `total_tokens`: Token counts
- `estimated_cost_usd`: Cost estimate based on configured rates
- `models`: Per-model breakdown

Exposed in response headers: `X-Trace-ID`, `X-LLM-Total-Tokens`, `X-LLM-Estimated-Cost-USD`.

### Stage Timing

Each pipeline stage records its wall-clock duration in `ctx.stage_timings_ms`. A structured `TRACE` JSON line is emitted:

```json
{
  "trace_id": "abc-123",
  "session_id": "def-456",
  "stage": "stage_0_5_router_match",
  "duration_ms": 3.42,
  "extra": { "matched": true }
}
```

### Structured Trace Events

Detailed decision data is emitted via `trace_detail()` with `TRACE_DETAIL` prefix:

| Stage | Events | Key data |
|-------|--------|----------|
| stage_0_2_question_analyzer | validated, artifact, error | query_type, preferred_path, confidence, candidate_tools |
| stage_0_5_router_match | (via _trace_stage) | matched (bool) |
| stage_0_7_analyzer_route | decision | invocation_built, preferred_path, top_tool, top_score |
| stage_1_generate_plan | plan_ready, artifact | plan dict, raw_sql_present, question_analysis_used |
| stage_2_sql_execute | sql_ready, artifact, blocked, sql_result, error | sql_hash, tables, rows, cols, elapsed_ms |
| stage_3_analyzer_enrich | enrichment_ready, why_analysis | share_override, why_override, correlation_keys |
| stage_4_summarize_data | provenance_gate, summary_ready, final_summary | gate_passed, coverage, confidence, claims_count |
| stage_5_chart_build | selection | chart_type, row_count, reason |

### Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /healthz` | Liveness probe — returns `{"status": "ok"}` |
| `GET /readyz` | Readiness probe — checks DB connection + schema reflection |
| `GET /metrics` | Full metrics snapshot: request counts, LLM usage, tool stats, cache stats, circuit breaker state, per-stage timing averages |

### Metrics Inventory (34+ counters)

**Request:** request_count, total_request_time, error_count
**LLM:** llm_call_count, prompt/completion/total_tokens, estimated_cost_usd, per-model breakdown
**SQL:** sql_query_count, total_sql_time
**Tools:** tool_call_count, tool_error_count, total_tool_time
**Agent:** agent_round_count, data/conceptual/fallback_exit_count
**Guardrails:** firewall_allow/warn/block_count, summary_schema/grounding_failure_count, provenance_gate_failure_count, relevance_block_count
**Router:** deterministic/semantic/analyzer_match_count, miss_count, fallback_intents
**Resilience:** circuit_open_events, load_shed_count
**Security:** security_event_count, security_events_by_type

---

## 8. Conversation History Lifecycle

End-to-end flow of conversation context:

```
1. Frontend (ChatPage.jsx)
   └─ User sends message
   └─ After response: record_chat_turn_txn RPC saves Q+A to chat_history table

2. Edge Function (chat-with-enerbot)
   └─ Loads last 6 rows from chat_history (user_id filter, newest first)
   └─ Reverses to chronological order
   └─ Pairs user→assistant messages via sequential scan
   └─ Sends as conversation_history in request body (max 3 pairs)

3. Railway Backend (main.py)
   └─ Resolves session_id from X-Session-Token header
   └─ Checks in-process session store for existing history
   └─ If empty + edge function provided history:
       └─ Truncates items to 2000 chars each
       └─ Seeds session store
   └─ Passes bound_history to process_query()

4. Pipeline Consumption
   └─ Stage 0.2 (Question Analyzer): history enables follow-up resolution
       └─ "What about 2024?" → analyzer sees previous "prices in 2023" context
   └─ Stage 4 (Summarizer): history provides conversational context
   └─ Stage 0.5 (Keyword Router): NO access to history (stateless, query-only)
       └─ This is why Stage 0.7 exists — the analyzer fallback fills this gap
```

**Key constraint:** The keyword router (Stage 0.5) is stateless by design — it only sees the current query. Conversation continuity depends on the LLM Question Analyzer (Stage 0.2/0.7), which receives the full conversation history. When a follow-up query like "What about 2024?" arrives, the keyword router misses (no price/tariff keywords), but the analyzer correctly identifies it as a continuation and routes to the appropriate tool.
