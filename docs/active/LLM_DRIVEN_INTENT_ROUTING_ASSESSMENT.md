# Architectural Assessment: LLM-Driven Intent Routing Proposal

Date: 2026-03-15

## Context

This document assesses the proposal to replace keyword-based query routing with an LLM-driven intent routing layer ("The Decoupled Pipeline"). The assessment is based on a thorough audit of the existing codebase to determine what already exists, what gaps remain, and what work is actually needed.

---

## Executive Summary: You're Closer Than You Think

**The proposed 5-stage architecture is ~80% already implemented.** The codebase already has a sophisticated multi-stage decoupled pipeline with security firewalls, heuristic bypass, LLM-based question analysis, deterministic tools, agent orchestration, provenance-gated summarization, and server-signed conversational memory. The proposal accurately diagnoses real problems but describes a target architecture that largely mirrors what's already built.

**The real deltas are:**
1. **Fix the broken session token bridge** — Railway's session memory exists but the edge function (`chat-with-enerbot`) never forwards `X-Session-Token`, so every request gets a fresh empty session. Conversation history is effectively dead.
2. **Promote the existing LLM Question Analyzer** from optional hint-provider to primary router, feeding the (now-working) conversation history into the routing decision.
3. **Add per-stage model configuration** so routing uses cheap/fast models while summarization uses more capable ones.

---

## Stage-by-Stage Gap Analysis

### Stage 1 (Proposal) — "AI Firewall" → ALREADY EXISTS

**Current implementation:** `langchain_railway/guardrails/firewall.py`
- Regex-based prompt injection detection (instruction override, exfiltration, role hijack)
- Dangerous SQL intent blocking (DROP, TRUNCATE, ALTER, INSERT, DELETE, UPDATE)
- Risk scoring system (block rules: +5 pts, warn rules: +1-2 pts, threshold: 8)
- Query sanitization (markdown stripping, control char removal, length cap at 1800 chars)
- Formal red-team gate with weighted scoring (`langchain_railway/guardrails/redteam_gate.py`)

**Gap:** None. This is fully operational and tested. No changes needed.

---

### Stage 2 (Proposal) — "Fast Heuristic Bypass" → ALREADY EXISTS

**Current implementation:** `langchain_railway/agent/planner.py` `prepare_context()`
- `is_conceptual_question()` — regex-based detection of definition queries ("What is a PPA?")
- `detect_analysis_mode()` — classifies light vs. analyst mode using keyword sets
- `detect_language()` — EN/KA/RU/ZH detection
- Conceptual short-circuit at `pipeline.py:108` bypasses all data stages

**The proposal's mention of "negative lookaheads on words like 'reason' or 'impact'"** — this nuance is partially handled. The `is_conceptual_question()` function catches simple "What is X?" patterns, but the question analyzer (Stage 0.2) provides the more nuanced classification (e.g., distinguishing `conceptual_definition` from `data_explanation`).

**Gap:** Minor. The heuristic bypass works but could benefit from the negative lookahead refinements mentioned in the proposal for edge cases like "What is the reason for the price increase?" Currently these are caught downstream by the question analyzer, not the fast-path regex. This is a low-priority enhancement — the system handles it correctly, just not at the cheapest possible layer.

---

### Stage 3 (Proposal) — "LLM Intent Gatekeeper" → PARTIALLY EXISTS (This is the main gap)

**Current implementation:** Two separate systems that don't fully integrate:

**System A — Keyword Router** (`langchain_railway/agent/router.py` `match_tool()`):
- 4 deterministic keyword chains (tariff, composition, generation, price)
- Semantic fallback with competitive scoring
- Parameter extraction (dates, entities, metrics, currency)
- **CRITICAL LIMITATION: Takes only `(query, is_explanation)` — zero access to conversation history**

**System B — LLM Question Analyzer** (`langchain_railway/agent/planner.py` → `langchain_railway/core/llm.py` `llm_analyze_question()`):
- Receives query + conversation_history + domain catalogs
- Returns structured `QuestionAnalysis` JSON with intent, tool candidates, parameters, routing preference
- Operates in "shadow" (logging) or "active" (hints) mode
- **Currently only provides hints — does NOT replace the deterministic router**

**The exact problem the proposal describes is real:**
1. User asks: "Show me balancing prices in 2023" → `match_tool()` returns `get_prices` ✓
2. User follows up: "What about 2024?" → `match_tool()` sees no price keyword → returns `None` ✗
3. The keyword router evaluates each query in isolation because it has no history access

**The LLM Question Analyzer already solves this** — it receives history and can contextualize "What about 2024?" — but its output is currently consumed only as optional hints downstream, not as the authoritative routing decision.

**Gap: This is the primary work item.** The routing decision must be restructured so:
1. The LLM Question Analyzer becomes the primary router for non-trivial queries
2. The keyword router is preserved as a fast-path optimization (high-confidence, unambiguous queries)
3. When keyword router returns `None` or low confidence, the LLM analyzer's `routing.preferred_path` and `candidate_tools` become the authoritative routing decision

---

### Stage 4 (Proposal) — "Deterministic Tools & Agent Execution" → ALREADY EXISTS

**Current implementation:**

**Typed Tools** (`langchain_railway/agent/tools/`):
- `get_prices` — balancing/deregulated/guaranteed-capacity prices, exchange rates
- `get_tariffs` — Enguri, Gardabani TPP, old TPP group tariffs
- `get_generation_mix` — generation/demand by technology type
- `get_balancing_composition` — balancing market entity shares
- All use parameterized SQL with `run_text_query()` (no LLM-generated SQL)
- Tool registry at `langchain_railway/agent/tools/registry.py`

**Agent Loop** (`langchain_railway/agent/orchestrator.py` `run_agent_loop()`):
- ReACT-style bounded loop (max 3 rounds, configurable via `AGENT_MAX_ROUNDS`)
- LLM bound with tool schemas, restricted to `ALLOWED_AGENT_TOOL_NAMES` whitelist
- Three exit conditions: `data_exit`, `conceptual_exit`, `fallback_exit`
- Currently disabled by default (`ENABLE_AGENT_LOOP=false`)

**Ad-hoc SQL Fallback** (`langchain_railway/agent/sql_executor.py`):
- LLM generates plan + SQL via `planner.generate_plan()`
- SQL validated: sanitization, table whitelist, synonym repair, LIMIT enforcement
- Read-only transaction enforcement, 30-second timeout
- Circuit breakers for DB failures

**Python Analysis** (`langchain_railway/agent/analyzer.py` — 1381 lines):
- Correlation analysis (Pearson), trend calculations, share shifts
- CAGR forecasting, YoY comparisons, seasonal pattern detection
- All math done in Python — zero LLM math

**Gap:** None structurally. The agent loop being disabled by default (`ENABLE_AGENT_LOOP=false`) is a deployment decision, not a code gap. The proposal's vision of multi-step agent execution is fully implemented and waiting to be turned on.

---

### Stage 5 (Proposal) — "Grounded Summarization & Provenance" → ALREADY EXISTS

**Current implementation:** `langchain_railway/agent/summarizer.py`

**Strict Grounding:**
- Summarizer LLM receives only pre-computed data + domain knowledge
- `strict_grounding=True` retry when initial response fails grounding check
- Response schema: `SummaryEnvelope(answer, claims, citations, confidence)`

**Provenance Gate** (`_enforce_provenance_gate()`):
- Extracts all numeric tokens from LLM claims
- Maps each token to exact database cell references (row_index, column, value, SHA256 fingerprint)
- Coverage = grounded_numeric_claims / total_numeric_claims
- Gate threshold: `PROVENANCE_MIN_COVERAGE=1.0` (100% required by default)
- On failure: answer completely replaced with conservative fallback, confidence dropped to 0.2

**Gap:** None. This is the most robust part of the system.

---

### Conversational Memory → EXISTS BUT BROKEN END-TO-END

**Railway side** (`langchain_railway/utils/session_memory.py`):
- HMAC-SHA256 signed session tokens, in-process thread-safe storage
- Last 3 Q&A turns stored per session (`SESSION_HISTORY_MAX_TURNS=3`)
- `main.py:898` reads `X-Session-Token` header, issues new if missing
- `main.py:963` returns `X-Session-Token` in response header
- `main.py:976` retrieves `bound_history` from session store
- `main.py:1023` passes `bound_history` to `process_query()`
- `main.py:1040` appends Q&A exchange after successful response

**Frontend side** (`D:\export_enai\src\pages\ChatPage.jsx`):
- Chat history stored in Supabase `chat_history` table (last 20 messages loaded on page init)
- Persistence via `record_chat_turn_txn` RPC after each response
- Database-enforced auth, quota, and RLS

**The bridge (edge function) is broken** (`D:\export_enai\edge_functions\chat-with-enerbot.txt`):
- Sends to Railway: `{ query, service_tier }` with headers `X-App-Key`, `X-Request-Id`
- **Does NOT forward `X-Session-Token` to Railway**
- **Does NOT read `X-Session-Token` from Railway's response**
- **Does NOT send conversation_history in the request body**

**Consequence:** Every request to Railway arrives without a session token → Railway issues a fresh session → `get_history(session_id)` returns `[]` → the question analyzer and summarizer always receive empty history → **conversational memory is effectively dead in production**, even though the mechanism is fully coded on both sides.

**This is the #1 prerequisite** before any LLM routing upgrade can deliver on the "conversational memory" promise.

**Two-part gap:**
1. The session token must flow: browser <-> edge function <-> Railway (or conversation history must be forwarded)
2. History reaches the question analyzer and summarizer, but **NOT** the keyword router

---

## What Actually Needs to Change

### Pre-requisite: Fix the Session Token Bridge (Without This, Nothing Else Matters)

The entire proposal hinges on "the LLM sees the history." But history never arrives because the edge function drops the session token. This must be fixed first.

#### 0. Wire `X-Session-Token` through the edge function

**In `D:\export_enai\edge_functions\chat-with-enerbot`:**
- Forward the `X-Session-Token` header from the frontend to Railway (if present)
- Read `X-Session-Token` from Railway's response and return it to the frontend

**In `D:\export_enai\src\pages\ChatPage.jsx`:**
- Store the `X-Session-Token` received from Railway (via the edge function response)
- Send it back on subsequent requests in the same browser session

**Alternative approach (simpler, no session token):** Instead of wiring the session token, the edge function could load the last N chat turns from the `chat_history` table (it already has the authenticated user_id) and include them as `conversation_history` in the request body to Railway. This leverages the existing database persistence and avoids the complexity of managing session tokens across the edge function boundary. Railway already has code to accept (but currently ignore) `conversation_history` from the request body — `main.py:970` logs it as `client_history_ignored`. This would need to be changed to trust edge-function-provided history (since it comes from a server-side authenticated context, not the browser).

### The Core Change: Promote LLM Question Analyzer to Primary Router

The architectural change is smaller than the proposal suggests. Rather than building a new "LLM Intent Gatekeeper," the existing `llm_analyze_question()` needs to be promoted from hint-provider to authoritative router. Here's the concrete work:

#### 1. Restructure routing priority in `langchain_railway/agent/pipeline.py`

**Current flow (lines 114-250):**
```
keyword router (primary) → agent loop (fallback) → SQL (last resort)
```

**New flow:**
```
keyword router (fast-path, high-confidence only)
  → IF matched with confidence >= threshold: execute tool
  → ELSE: LLM question analyzer (primary router, history-aware)
    → IF preferred_path=tool + candidate identified: execute tool
    → IF preferred_path=knowledge: conceptual answer
    → IF preferred_path=sql: SQL fallback
    → IF preferred_path=clarify: ask for clarification
    → ELSE: agent loop → SQL fallback
```

#### 2. Extract tool parameters from QuestionAnalysis

The `QuestionAnalysis` contract already returns `candidate_tools` with `params_hint`. A new function is needed to convert these hints into a `ToolInvocation` with fully resolved parameters (dates, entities, metrics). The existing `_extract_date_range()`, `_extract_currency()`, etc. from `router.py` should be reused for parameter extraction, supplemented by the analyzer's `sql_hints` (which contain `period`, `entities`, `metrics`).

#### 3. Add per-stage model configuration via Railway env vars

**Current state:** Single `MODEL_TYPE`/`GEMINI_MODEL`/`OPENAI_MODEL` for all LLM calls.

**Needed:** The proposal asks for cheap/fast models for routing and more capable models for summarization. Add env vars:
```
ROUTER_MODEL=gemini-2.0-flash-lite    # Fast, cheap for intent classification
SUMMARIZER_MODEL=gemini-2.5-flash     # More capable for grounded summarization
PLANNER_MODEL=gemini-2.5-flash        # SQL generation
```

This requires modifying `llm_analyze_question()` and `llm_summarize_structured()` in `langchain_railway/core/llm.py` to accept a model override parameter, falling back to the global default.

#### 4. Enable agent loop by default

The proposal envisions multi-step tool execution as a core capability. `ENABLE_AGENT_LOOP` should flip from `false` to `true` now that it has been validated. This is a one-line config change.

---

## Files to Modify

### Railway backend (`D:\Enaiapp\langchain_railway\`)

| File | Change | Complexity |
|------|--------|------------|
| `agent/pipeline.py` | Restructure routing priority: keyword fast-path → LLM analyzer as primary router | Medium |
| `config.py` | Add per-stage model env vars (`ROUTER_MODEL`, `SUMMARIZER_MODEL`, `PLANNER_MODEL`), flip `ENABLE_AGENT_LOOP` default | Low |
| `core/llm.py` | Accept model override in `llm_analyze_question()`, `llm_summarize_structured()`, `llm_generate_plan_and_sql()` | Medium |
| `agent/planner.py` | New `build_tool_invocation_from_analysis()` — convert QuestionAnalysis → ToolInvocation | Medium |
| `agent/router.py` | No structural changes — extract helper functions for reuse by the new invocation builder | Low |
| `main.py` | If using the alternative approach: trust edge-function-provided `conversation_history` instead of ignoring it (needs a flag or header to distinguish edge-function origin from raw browser) | Low-Medium |

### Frontend / Edge function (`D:\export_enai\`) — session bridge fix

| File | Change | Complexity |
|------|--------|------------|
| `edge_functions/chat-with-enerbot` | Either: (A) forward `X-Session-Token` header both ways, or (B) load last N chat turns from `chat_history` table and include as `conversation_history` in request body | Low-Medium |
| `src/pages/ChatPage.jsx` | If approach A: store and resend `X-Session-Token` from response headers | Low |

**No changes needed to:** firewall, tools, tool registry, orchestrator, sql_executor, analyzer, summarizer, provenance gate, session_memory, contracts, database schema.

---

## Risk Assessment

### Low Risk
- **Security:** Firewall remains Stage 0, unchanged. No new attack surface.
- **Provenance:** Gate unchanged. 100% numeric grounding still enforced.
- **Tools:** Deterministic tools unchanged. No new SQL generation paths.
- **Rollback:** Feature flags already exist. If LLM routing underperforms, set `ENABLE_QUESTION_ANALYZER_HINTS=false` and the system reverts to keyword routing.

### Medium Risk
- **Latency:** Adding an LLM call to the routing path adds ~200-500ms. Mitigated by: (a) keyword router as fast-path for unambiguous queries, (b) using a fast/cheap model (Flash Lite), (c) the question analyzer is already being called in active mode — this just moves its influence earlier.
- **Cost:** Each routed query now consumes LLM tokens for classification. Mitigated by: (a) response caching already in place (hash on query + history + catalogs), (b) conceptual queries still short-circuit before LLM routing, (c) Flash-tier models are very cheap (~$0.01-0.02 per 1K routing calls).

### Considerations
- **The keyword router should NOT be removed.** It handles >80% of queries instantly at zero cost. The LLM router should only activate when the keyword router returns `None` or low confidence. This hybrid approach is cheaper and faster than sending every query through an LLM.
- **Shadow mode validation first.** Before promoting the analyzer to primary router, run both paths in parallel (shadow mode already exists) and measure disagreement rates. Only promote when the analyzer matches or beats keyword routing on the existing query distribution.

---

## Proposal Accuracy Scorecard

| Proposal Claim | Status | Notes |
|---|---|---|
| Current routing is keyword-based and brittle | **Accurate** | `match_tool()` uses hardcoded keyword chains |
| Lacks conversational memory in routing | **Accurate — worse than expected** | Not just routing — session memory is dead end-to-end because edge function never forwards `X-Session-Token`. History is always `[]` for every request. |
| Semantic confusion with overlapping keywords | **Accurate** | Competitive scoring helps but doesn't eliminate |
| Stage 1: AI Firewall needed | **Already exists** | `firewall.py` + `redteam_gate.py` |
| Stage 2: Fast heuristic bypass needed | **Already exists** | `prepare_context()` + conceptual short-circuit |
| Stage 3: LLM Intent Gatekeeper needed | **Partially exists** | `llm_analyze_question()` exists but is hint-only |
| Stage 4: Deterministic tools + agent loop | **Already exists** | 4 typed tools + ReACT orchestrator |
| Stage 5: Grounded summarization + provenance | **Already exists** | 100% provenance gate operational |
| Use cheap models for routing, better for summary | **Not yet implemented** | Single model config currently |
| Railway env vars for model management | **Partially exists** | Global model config exists, per-stage does not |

---

## Recommended Implementation Order

0. **Fix session token bridge** in edge function + frontend (prerequisite — without this, history is always empty)
1. **Add per-stage model env vars** in config.py + model override in llm.py (low risk, immediate value)
2. **Build `build_tool_invocation_from_analysis()`** in planner.py (converts QuestionAnalysis → ToolInvocation)
3. **Run shadow comparison** — log keyword router vs. analyzer routing decisions for 1-2 weeks
4. **Restructure pipeline.py routing** — keyword fast-path + LLM analyzer as primary fallback
5. **Flip `ENABLE_AGENT_LOOP=true`** after validating routing accuracy

## Verification Plan

0. **Session bridge test:** Make two sequential requests through the edge function. Verify Railway receives a valid `X-Session-Token` on the second request and `bound_history` is non-empty. Check Railway logs for `session_bound_history_turns > 0`.
1. **Unit tests:** Test `build_tool_invocation_from_analysis()` with the examples from `langchain_railway/skills/question-analyzer/references/examples.md`
2. **Conversation continuity test (end-to-end):** Via the chat UI — "Show me balancing prices in 2023" → "What about 2024?" — verify second query routes to `get_prices` with updated dates
3. **Semantic disambiguation test:** "What is GNERC?" (→ conceptual) vs. "What is the reason for the price change?" (→ data_explanation) — verify different routing paths
4. **Cost regression:** Compare per-query LLM token usage before/after on a representative query set
5. **Latency regression:** Measure p50/p95 response times before/after, ensuring keyword-matched queries show no regression
6. **Red-team gate:** Re-run `evaluate_redteam_gate()` to confirm security posture unchanged
