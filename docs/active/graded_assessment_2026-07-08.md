# Graded Code/App Assessment — Enai AI Analyst (`langchain_railway`)

**Date:** 2026-07-08 (post agent-loop deletion, architecture-audit P0/P1 series, efficiency pass, and the six design-gap closures)
**Scale:** F–A per dimension, same rubric as [`graded_assessment_2026-06-10.md`](graded_assessment_2026-06-10.md): A = exemplary for a production service of this size; B = solid with known, managed gaps; C = functional but with material weaknesses; D = significant deficiency; F = broken/absent.
**Basis (all executed during this assessment):** targeted suite **1190 passed** with coverage (**76%** vs 70% floor), security suite **19 passed**, red-team gate **score 1.0, pass**, `ruff check .` (9 violations found → fixed → clean), dependency/CI/workflow inspection, and focused reads of the auth/rate-limit stack, resilience layer, SQL validation stack, and session store. Baseline for comparison: the 2026-06-10 assessment (overall B-).

| Dimension | 2026-06-10 | **2026-07-08** |
|---|---|---|
| Architecture | B- | **B+** |
| Code quality | C+ | **B-** |
| Security | B- | **B** |
| Performance | C+ | **B-** |
| Tests / Build / CI | B- | **A-** |
| **Overall** | **B-** | **B** |

---

## Findings registry (bugs and inefficiencies)

Ranked by severity. "Fixed" = fixed and committed during this assessment.

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | **CI lint gate was red**: 9 ruff violations (4 pre-dating this session — `analyzer.py`, `metric_config.py`, `vector_store.py`; 5 from the design-gap series, incl. a duplicate `import os` F811) | CI-blocking | **Fixed** (`a47aa90`) |
| 2 | **Pre-auth rate limiter keys on the socket peer** (`get_remote_address`) while the gateway limiter correctly parses `X-Forwarded-For` ([main.py:410](../../main.py) vs :370). Behind the Railway proxy every external caller shares one pre-auth bucket: one abusive client can 429 everyone, and the per-caller limit is meaningless. Carried open since June. | Moderate (availability) | Open — one-line fix, but confirm Railway edge XFF semantics first (operator) |
| 3 | **Gateway-mode 500s still leak exception text** (`detail=f'Query processing failed: {e}'`, [main.py:1064](../../main.py)); bearer mode is sanitized one line above. June S8, unchanged. Exposure is internal-only (the edge function sanitizes onward), but the asymmetry is unjustified. | Low-moderate (info leak) | Open — trivial fix |
| 4 | **Rate-limit buckets are never evicted**: pruned-empty subject keys stay in the dicts forever; key count grows with distinct IPs/sessions for the process lifetime. | Low (slow memory growth; resets on deploy) | Open — delete key when its window empties |
| 5 | **`psycopg2-binary` is a dead dependency**: zero importers; the engine URL is always coerced to `postgresql+psycopg://` (v3) by `core/db.py`. Container weight + supply-chain surface for nothing. | Low | Open — remove from requirements, verify Docker build |
| 6 | **Threshold patterns miss spelled-out "percent"** (`_SHARE_THRESHOLD_PATTERNS` require a literal `%`), and are English-only — KA/RU threshold phrasings never match. Documented in architecture §5.9. | Low (routing consistency) | Open — "% or percent" is a small gated fix; multilingual variants should come from harvested production fixtures |
| 7 | **DB pool (5) vs backpressure (8) mismatch**: under 6–8 concurrent data requests, the excess stalls up to 30s on connection checkout, mis-attributed to tool latency in traces. | Moderate under load (invisible today) | Open — operator decision (needs Supabase/PgBouncer budget) |
| 8 | **Coverage is thinnest exactly where risk is thickest**: `core/llm.py` 52% (766 uncovered lines), `main.py` 60% (402), `llm_runtime.py` 63%, `chart_selector.py` 55%. The 76% total is honest but unevenly distributed. | Structural | Open — target the big-4 files when raising the floor |
| 9 | **LIGHT-tier vector retrieval still serial** ahead of tool execution (~0.3–0.6s hideable per narrative data answer); designed as a flag-gated follow-up. | Low-moderate (latency) | Open — planned |
| 10 | Naive `datetime.now().year` in `analyzer.py:2598` (server-local year at TZ boundaries) | Cosmetic | Open — harmless in practice |

Non-findings verified clean during the hunt: no mutable default arguments; the only `except: pass/continue` sites are narrowly typed coercion guards; session-store cleanup runs per request under lock (bounded); no raw-`postgresql://` engine creation anywhere; version strings unified; only 4 TODO markers in ~39.6k production lines.

---

## Architecture — B+ (was B-)

**Up since June:**
- The three stacked legacy layers June criticized are down to one: the **agent loop is deleted** (audit P2), the SQL escape hatch remains as the single documented fallback. `process_query` is a thin `StageResult` driver (P0-4).
- The "monologue" design gaps are closed or gated: answer provenance surfaced to clients, deterministic renders under shadow fitness checks, a self-growing routing golden set, contract continuity and evidence re-analysis implemented behind default-OFF flags with **pre-committed cutover criteria** (§5.7/§5.8) — the discipline June asked for ("gated on an open-ended condition with no owner/date") is now the house pattern.
- Ontology migration has a measured path (§5.9 agreement counters) instead of a hand-wave.
- The architecture document is now genuinely truthful: three drift passes this series, module map matches the tree, historical vs living sections separated.

**Holding it from A-:**
- Stage 0.7's four-strategy picker still awaits the same production-counter pull it awaited in May (F.6) — the instrument exists; the reading has never been taken.
- The planner still invents evidence steps (ontology dual-ownership) until §5.9 slices land.
- Single-replica in-process state is a documented, deliberate constraint — but it is still a hard scaling wall, and the new session-stored contract snapshots deepen the dependence.

## Code quality — B- (was C+)

**Up:** `core/llm.py` 4,253 → 3,340 with three clean extractions; dead modules actually deleted (`orchestrator`, `tool_adapter`, orphan renderer, 5 dead config constants); lint enforced in CI **and now actually green**; new modules ship with narrow responsibilities, never-raise contracts for observability code, and tests in the same commit; docstrings still cite motivating traces.

**Down:** `agent/analyzer.py` grew to 3,415 (Q4 split still deferred) and `pipeline.py` to 2,832 — the two highest-churn files are the two biggest; the multilingual keyword tables are only partially consolidated (A3.d: `query_validation.py`, `router.py` still hand-maintained); `summarizer.py` at 2,705 still mixes grounding engine + dispatch + formatters (the planned `grounding.py` extract).

## Security — B (was B-)

**Up:** everything June graded up still holds (DB least-privilege role, AST/whitelist/read-only SQL stack, firewall + signed sessions + fail-closed startup), and the red-team gate now also enforces **grounding detect/accept rates** — score 1.0 across the board today; 19 security tests green.

**Down / open:** findings #2 (shared pre-auth bucket — the one with real availability teeth), #3 (S8 error-detail leak, internal-only), #4 (bucket growth). None is new; all three are cheap; the fact they've survived two assessments is itself the finding.

## Performance — B- (was C+)

**Up:** Stage 0.8 tool calls prefetch concurrently; the embedding client is a reused singleton with a query memo (was: rebuilt per request); the analyzer thinking budget halved to 1024 (env-restorable); latency remains *engineered* — budgets, tiers, caches, per-stage timings.

**Down:** the two serial LLM calls remain the floor (irreducible without quality risk); LIGHT-tier retrieval overlap designed but not landed (#9); pool/backpressure mismatch (#7) is the hidden cliff under concurrency; the whole service is still sync-on-threadpool with `max_concurrent=8` — fine at current scale, a known ceiling beyond it.

## Tests / Build / CI — A- (was B-)

**Up:** 1,190 targeted + 19 security tests, all executed green today; CI now runs **ruff + security suite + scored red-team gate (with grounding thresholds) + full suite with a 70% coverage floor** (actual 76%), with pip caching — June's #1 and #4 recommendations, done. Beyond CI: a routing golden-set eval with fixture-contract tests, a production-failure→fixture harvest loop, and regression tests that cite their motivating traces.

**Holding it from A:** no integration tier against a real database; no load/latency regression check; coverage concentrated away from `llm.py`/`main.py` (#8); and the lint gate being silently red until today suggests branch-protection/required-checks aren't enforced on this repo — worth turning on.

---

## Highest-leverage moves (ranked)

1. **Fix the pre-auth limiter key** (#2) — one line reusing the existing XFF parse, after confirming Railway edge semantics. The only finding with real production-incident potential.
2. **Turn on branch protection with required CI checks** — the red lint gate proved merges can bypass CI verdicts.
3. **Sanitize the gateway 500 detail** (#3) and **evict empty rate buckets** (#4) — fifteen-minute hygiene pair.
4. **Drop `psycopg2-binary`** (#5) — dependency diet, verify container build.
5. **Take the Stage 0.7 reading** — pull 14 days of `stage_0_7_*` counters; the architecture's oldest open decision is one query away.
6. When raising the coverage floor, **spend it on `core/llm.py` and `main.py`** — the uncovered 1,168 lines there are where the next regression hides.

*Assessment performed read-mostly; the only code changed was the lint remediation (`a47aa90`), committed separately.*
