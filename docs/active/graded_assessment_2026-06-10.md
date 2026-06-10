# Graded Code/App Assessment — Enai AI Analyst (`langchain_railway`)

**Date:** 2026-06-10 (post medium-fix implementation: P1–P4 applied; P5 skipped by owner decision)
**Scale:** F–A per dimension. Grading rubric: A = exemplary for a production service of this
size; B = solid with known, managed gaps; C = functional but with material weaknesses; D =
significant deficiency; F = broken/absent.
**Basis:** full read of entry points, SQL safety layer, guardrails, tools, contracts, CI,
Docker, tests (executed: 1052 targeted + 19 security, all green; redteam gate score 1.0),
plus the prior audit ([`audit_2026-06-10.md`](audit_2026-06-10.md)) and fix series
([`medium_fix_plan_2026-06-10.md`](medium_fix_plan_2026-06-10.md)).

| Dimension | Grade |
|---|---|
| Architecture | **B-** |
| Code quality | **C+** |
| Security | **B-** |
| Performance | **C+** |
| Tests / Build / CI | **B-** |
| **Overall** | **B-** |

*(Status 2026-06-10: the P1–P4 fix series is committed and pushed to `main` (commit
`3b3b77a`); the Supabase least-privilege role is applied and its password has been rotated by
the operator. The earlier deployment caveat is resolved.)*

---

## Architecture — B-

**What earns the grade up:**
- Contract-driven pipeline with a single interpretation point (Stage 0.2 emits the full
  answer contract) and deterministic rendering for five answer shapes — a real, working
  design, not aspiration. Evidence frames, typed tools, and the generic renderer give new
  question families a no-code path.
- Clean module decomposition (`agent/`, `core/`, `contracts/`, `knowledge/`, `guardrails/`)
  after a documented 6-phase extraction from a monolith; `QueryContext` as the single mutable
  state object is simple and traceable.
- Unusually disciplined architecture doc that names code as source of truth and tracks its own
  debt (§5), with invariants pinned to tests.
- Resilience primitives where they matter: circuit breakers (LLM, DB), backpressure gate,
  graceful degradation of SQL-validation failures to conceptual answers.

**What holds it down:**
- Three stacked legacy fallback layers (agent loop, 4-strategy primary picker, SQL escape
  hatch) still live under the "contract executes" story; Stage 0.7 removal is gated on an
  open-ended production-data condition with no owner/date — effectively permanent.
- "One LLM call" framing oversells determinism (analyzer + summarizer + forecast path +
  provider fallback are all LLM calls); the doc itself is internally inconsistent on the
  "single execution loop" claim (see prior audit §A1–A3).
- The central bet — trust the analyzer contract — has no deterministic backstop for
  "right shape, wrong tool/metric" misrouting; mitigations are heuristics checking heuristics.
- In-process state (sessions, rate limits, caches) silently assumes a single worker/replica;
  the architecture has no stated scaling story (P5 was declined — acceptable, but the
  constraint should be pinned in deployment docs).

---

## Code quality — C+

**Up:**
- Typed tools and contracts are genuinely good: enum-validated params, bind parameters,
  Pydantic models with a JSON-schema snapshot asserted in tests.
- Docstrings are consistently informative and often cite the production trace/commit that
  motivated a change — rare and valuable.
- Ruff is configured (`pyproject.toml`) with sensible first-party isort groups.

**Down:**
- File-size pathology: `core/llm.py` 4,253 lines, `agent/analyzer.py` 3,285,
  `agent/pipeline.py` 2,558, `agent/summarizer.py` 2,461. Four files ≈ 12.5k lines ≈ 20% of
  the codebase; each mixes multiple responsibilities (prompt assembly + caching + provider
  fallback + parsing in one module).
- Heuristic sprawl: large hand-tuned regex/keyword tables across three languages duplicated in
  at least four places (`query_validation`, `router`, `planner`, `models.QueryContext`
  property logic) — the exact brittleness the architecture claims to have escaped lives on in
  fallback paths and will drift.
- Hygiene debt: `from config import *` in `main.py`; deprecated `@app.on_event("startup")`;
  re-export shims and intentionally-unused imports kept for backward compat; version strings
  disagree (`main.py v20.0` vs `FastAPI(version="18.6")`); orphan code acknowledged in docs
  (`generic_renderer._render_forecast`); duplicated share-pivot SQL builders
  (`BALANCING_SHARE_PIVOT_SQL` vs `build_trade_share_cte`) that must be kept in sync by hand.
- Lint is configured but not installed locally and **not enforced in CI** — so the standard
  exists on paper only.

---

## Security — B- (code) / with the deployment caveat above

**Up (much of this is the 2026-06-10 fix series):**
- **DB privilege boundary now real (P4, live):** API connects as a role with SELECT on exactly
  the whitelisted relations; `auth.*` and writes fail at the database regardless of app-layer
  bugs. This converts the whole SQL stack into defense-in-depth.
- Layered SQL safety: AST-based whitelist (sqlglot), single-statement enforcement (P1),
  SELECT-only root check, forbidden-node scan, function deny/allow lists, `SET TRANSACTION
  READ ONLY`, statement timeout, row/memory caps.
- Multilingual firewall (P3) with a formal scored red-team gate in CI (block-rate 1.0,
  false-block 0.0); untrusted bearer-mode history is now firewalled (S6).
- Auth fundamentals sound: constant-time secret comparison, HMAC-signed session tokens, JWT
  with `aud`/`exp`/`sub` required, fail-closed startup validation of required secrets,
  env-gated + admin-gated `/metrics` and `/evaluate` (the latter blocked outside dev/test).
- Edge function (frontend boundary) enforces auth, account status, quota, and sanitizes
  backend errors.

**Down:**
- Multi-replica weakness stands (P5 declined): rate limits multiply by replica count;
  pre-auth limiter keys on socket peer behind a proxy (shared bucket); bucket keys never
  evicted.
- Gateway-mode error detail still leaks exception text (S8); referer check is security
  theater (S9); Georgian/Russian firewall phrasings flagged for native-speaker review.

---

## Performance — C+

**Up:**
- Latency is engineered, not accidental: per-stage prompt budgets with section-aware
  truncation, fast/deep pipeline modes, thinking-budget cap on the analyzer, three-tier vector
  retrieval (SKIP for deterministic paths), LLM response cache, catalog JSON serialized once
  at import.
- DB side is disciplined: materialized views, conservative pool sized for PgBouncer,
  `pool_pre_ping`/recycle, 30s statement timeout, incremental fetch with row cap, 100MB
  result-memory guard.
- Real observability: per-stage timings in every response, token/cost telemetry per request,
  trace events throughout — you can *see* where time goes.

**Down:**
- Latency is inherently LLM-dominated (analyzer + summarizer + enrichment ⇒ multi-second
  p50; the evaluate targets concede <8s simple / <45s complex). Nothing overlaps: evidence
  steps execute sequentially; secondary evidence, vector retrieval, and enrichment could
  parallelize but don't.
- Whole service is synchronous (`def` endpoints on a threadpool) with `max_concurrent=8` —
  fine for current scale, a ceiling beyond it; in-process caches reset on every deploy.
- Two Postgres drivers shipped (`psycopg2-binary` + `psycopg`) — one is dead weight; pinned
  2024-era langchain stack (0.1.x) forecloses cheap upgrades.
- No load test / latency budget regression check anywhere in CI.

---

## Tests / Build / CI — B-

The prior **D** is no longer defensible against the evidence (it likely predates visibility
into the suite): **1,052 targeted tests + 19 security tests, all green, executed during this
audit**, and CI that runs the security suite, the scored red-team gate, and the full suite on
every push/PR.

**Up:**
- Large, fast-by-design suite (~16 min serial) with a written policy: directory-sweep scope
  (fails closed for new files), no real LLM/DB calls in `tests/`, regression tests cite the
  production trace that motivated them.
- A *quantified* security gate in CI (`redteam_gate --min-score 0.92`) is something most
  projects of this size don't have.
- Deterministic builds: fully pinned requirements; simple working Dockerfile; schema snapshot
  test pins the LLM contract.

**Down:**
- **No coverage measurement at all** — 1,052 tests, unknown coverage; the giant modules
  (`llm.py`, `analyzer.py`) are precisely where untested branches hide.
- **No lint/type-check step in CI** (ruff configured but never run; no mypy/pyright) — the
  quality bar isn't enforced where it counts.
- Environment skew: CI and Docker on Python 3.11, local dev on 3.14, ruff targets py311 —
  works today, but version-specific breakage lands silently.
- No pip caching in CI (slow installs), redundant `pip install sqlalchemy`, no test-result
  artifacts, no parallelization (`pytest-xdist`), no integration tier against a real
  staging DB, no branch protection evident.

---

## Overall — B-

A coherent, well-documented, defensible system that has visibly improved through disciplined,
test-gated iteration — held back from B/B+ by code-size pathology in four core modules,
heuristic duplication on fallback paths, missing coverage/lint enforcement, and operational
gaps (uncommitted fixes, single-replica assumptions).

### Highest-leverage moves to raise grades
1. **Add `ruff check` + `pytest --cov` (with a floor) to CI** (Tests/CI → B, Code quality → B-).
2. **Split `core/llm.py` and `agent/analyzer.py`** along their existing seams — prompt
   assembly / provider client / parsing; enrichment dispatch / metric computations
   (Code quality → B-/B).
3. **Schedule the Stage 0.7 / agent-loop removal decision** with a date and the hit-rate
   query, or re-document them as permanent (Architecture → B).
4. **Pin the single-replica constraint** in deployment docs (or revisit shared state when
   scaling matters) (Architecture/Security hygiene).

*Read-only assessment; no code changed.*
