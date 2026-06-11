# Architecture & Code-Quality Improvement Plan

**Date:** 2026-06-10
**Source:** [`graded_assessment_2026-06-10.md`](graded_assessment_2026-06-10.md) (Architecture B-, Code quality C+)
**Goal:** Architecture B- → B, Code quality C+ → B-/B — without behavior change.
**Workflow:** every phase follows [`developer-phased-audit`](../../skills/developer-phased-audit/SKILL.md):
phase boundary → one artifact → mechanical verification → **targeted suite green**
(`python -m pytest tests/ --ignore=tests/security -q`, ~16 min run alone) → independent audit.
Refactor phases are **pure structural moves** verified by diff symmetry, the same discipline
as the §5.1 consolidation in [`query_pipeline_architecture.md`](query_pipeline_architecture.md).

**Standing constraints (owner preferences):**
- No new runtime infrastructure or services.
- General fixes over per-symptom patches.
- Behavior frozen: refactors must not change any answer, trace shape, or metric counter.

---

## The one risk that governs every refactor phase: the monkeypatch surface

Tests patch functions **by module path** (e.g. `redteam_gate` and `test_guardrails` patch
`core.llm._invoke_with_resilience`; evidence-planner tests patch
`agent.evidence_planner.execute_tool`). Moving a function to a new module breaks those patches
silently — the test still passes while patching a dead reference, or fails confusingly.

**Rule for every move:** before relocating any symbol, grep `tests/` for its qualified name.
If it is patched anywhere:
1. keep a **re-export in the original module** (import-from-new-home), **and**
2. make internal callers resolve it **through the original module attribute** (late binding
   via `core.llm._invoke_with_resilience(...)`-style lookup or an injectable `executor=`
   parameter — the pattern `evidence_planner.execute_remaining_evidence` already uses).

A phase that cannot satisfy this cheaply should leave that symbol where it is and move the
rest. The audit step of each phase explicitly re-runs the known patch-heavy files
(`test_guardrails.py`, `tests/security/test_redteam_gate.py`, `test_evidence_planner.py`).

---

## Phase overview (status updated 2026-06-10 evening)

| Phase | Track | Item | Status |
|---|---|---|---|
| A1 | Arch | Architecture-doc truth pass + single-replica constraint pinned | ✅ done |
| Q1 | Quality | Extract LLM provider/runtime layer → `core/llm_runtime.py` | ✅ done (reduced scope: patched orchestration symbols stayed in `core.llm` per the governing rule; see module docstrings) |
| Q2 | Quality | Extract JSON payload parsing/sanitization → `core/llm_payloads.py` | ✅ done |
| Q3 | Quality | Extract truncation engine → `core/prompt_budget.py` (Q3a) | ✅ Q3a done (`_enforce_prompt_budget` stayed in `core.llm` — patched + patch-dependent). **Q3b (analyzer prompt-block assembly, ~800 lines) deferred** — next candidate, same shim discipline. `core/llm.py`: 4,253 → ~3,300 lines |
| Q4 | Quality | Split share-analysis family out of `agent/analyzer.py` | ⏸ deferred (next session) |
| A2 | Arch | Delete orphan `generic_renderer._render_forecast` (+206 lines of forecast-only helpers, dispatch branch, doc refs) | ✅ done |
| A3 | Arch | Consolidate multilingual intent lexicons → `contracts/intent_lexicon.py` | ✅ partial: module seeded; `sql_executor`, `planner`, `models` migrated with HEAD-identity checks. **A3.d remaining:** `utils/query_validation.py`, `agent/router.py` (listed in the module docstring) |
| Q5 | Quality | `main.py` hygiene: explicit config imports (per-file-ignore deleted), lifespan handler, `__version__` unified | ✅ done |
| Q6 | Quality | Both balancing share-pivot SQL artifacts rendered from one spec | ✅ done — generated strings **byte-identical** to the former literals (verified vs git HEAD); the pre-existing pivot/CTE asymmetry (`share_total_hpp`) preserved and documented |
| A4 | Arch | Stage 0.7 / agent-loop removal decision | ⏸ **owner action**: pull 14 days of `stage_0_7_*` counters from production `/metrics`; decision rule pre-committed in §A4 below |
| A5 | Arch | Cross-check disagreement-rate observability | ✅ done — `analyzer_cross_check_events` counters + disagreement trace event |

Recommended order: **A1 → Q1 → Q2 → Q3 → Q4 → A2 → A3 → Q5 → Q6 → A5**, with **A4**
scheduled independently (it waits on production metrics, not on code).
Q1–Q3 are sequential (same file); everything else is parallelizable between them.

---

## Track A — Architecture

### A1 — Documentation truth pass (zero risk, do first)

**Boundary:** make `query_pipeline_architecture.md` internally consistent and honest; no code.

Fixes (from the audit critique §A1–A3, §A6–A7):
- Restate the design principle as "**one interpretation call**" — Stage 0.2 interprets once;
  the summarizer/forecast/fallback LLM calls exist and are listed.
- Reconcile §2.3 vs §3.6 vs §6 on the "single execution loop": the truthful claim is "one
  orchestration **function** with three passes." Say that.
- State explicitly that tool selection is a **four-strategy priority cascade**, not
  "deterministic via `candidate_tools[0]`", until F.6 lands.
- Document that the provenance gate covers **narrative answers only** — deterministic renders
  have no post-hoc numeric check (this is a known accepted gap, not an oversight).
- Note the retrieval-tier coupling: a Stage 0.2 misclassification to DETERMINISTIC silently
  skips knowledge retrieval (§A7 risk).
- **Pin the single-replica deployment constraint** (P5 declined): in-process rate limits,
  sessions, and caches assume one worker — one paragraph in the architecture doc +
  DEVELOPER_GUIDE.

**Done when:** doc matches code; reviewers can no longer find the §2.3/§3.6/§6 contradiction.

### A2 — Orphan-code removal

**Boundary:** delete `generic_renderer._render_forecast` (orphaned from production since
commit `ef048d6`, 2026-05-22 — only a frame-builder unit test exercises it) and the unit test
branch that exists solely to exercise it; sweep `generic_renderer.py` and `chart_frame_builder.py`
for branches reachable only from it. Update the two doc references.

**Risk:** Low — grep for callers first; production path already excludes FORECAST from the
generic renderer. **Done when:** symbol gone, suite green, docs updated.

### A3 — One multilingual intent lexicon

**Boundary:** the same English/Georgian/Russian keyword tables are hand-maintained in at
least four places and have already drifted:
- `utils/query_validation.py` (`_ANALYTICAL_KEYWORDS`, `_DATA_INTENT_PATTERN`,
  definition/topic maps),
- `agent/router.py` (tool keyword/semantic terms),
- `agent/planner.py` (`ANALYTICAL_KEYWORDS`, mode detection),
- `agent/sql_executor.py` (`DEMAND_INTENT_KEYWORDS` / `SUPPLY_INTENT_KEYWORDS`, added in P2),
- plus `models.QueryContext.analyzer_indicates_share_intent` inline phrase lists.

**Artifact:** `contracts/intent_lexicon.py` (or `utils/lexicon.py`) — a single module of named
frozen keyword sets (`ANALYTICAL`, `DATA_INTENT`, `DEMAND_SIDE`, `SUPPLY_SIDE`, `SHARE_INTENT`,
`DEFINITION`, …) with the three languages co-located per concept. Consumers import the sets;
**no matching-logic change** — the regexes/`any(...)` checks stay where they are, only the
word lists move.

**Sub-phases:** A3.a introduce module + migrate one consumer (`sql_executor`) → suite green;
A3.b–d migrate the rest one consumer per commit. Mechanical verification per consumer: the
compiled pattern/set is **string-identical** before and after (assert in a throwaway script,
or temporarily in a test).

**Risk:** Med — easy to subtly change a regex while lifting word lists; the string-identity
check is the guard. **Done when:** one canonical lexicon, every consumer imports it,
suite green.

### A4 — Stage 0.7 / agent-loop removal decision (data-gated)

**Boundary:** stop carrying "transitional" code indefinitely. This phase produces a
**decision**, not necessarily a deletion.

1. Confirm the F.2 counters (`stage_0_7_entered`, `stage_0_7_invocation_built`,
   `stage_0_7_used_result`) are visible in production metrics (they exist in code via
   `metrics.log_stage_0_7`).
2. Owner pulls 14 days of production numbers: `used_result / total requests`.
3. **Decision rule (pre-committed):** `< 5%` → delete strategies 3–4 from
   `_pick_primary_invocation` + associated traces/counters (small diff, listed in
   `query_pipeline_architecture.md` §5.2); `≥ 5%` → re-document Stage 0.7 as **permanent**
   and remove the "transitional" framing (A1 doc edit).
4. Same exercise for the legacy agent loop (`ENABLE_AGENT_LOOP` path): measure how often it
   fires (`agent_outcome != ""` with no authoritative analyzer) and either schedule removal
   or declare it permanent.

**Done when:** a dated decision is recorded in the architecture doc and, if the data clears,
the deletion ships behind a green suite. *(Code effort S; the long pole is production data —
schedule with the owner now.)*

### A5 — Cross-check disagreement observability

**Boundary:** the audit's standing criticism (§A5): mitigations for analyzer misrouting are
heuristics-checking-heuristics with no feedback loop. Cheapest general fix: **measure**.

**Artifact:** counters (existing `metrics` class, no new infra):
`analyzer_cross_check_disagreement` (LLM vs query_type-derived answer_kind),
`analyzer_cross_check_override_applied`, `legal_list_exception_applied`, and
`scenario_override_gated`. Plus one trace field carrying both candidate values on
disagreement.

**Why:** turns §5.3 threshold tuning (0.8 / 0.85 / 0.3 magic numbers) from intuition into
data, and feeds A4-style decisions later. **Done when:** counters land, visible in
`/metrics`, suite green.

---

## Track Q — Code quality

### Q1–Q3 — Decompose `core/llm.py` (4,253 lines → ~4 modules)

The file already has clean internal seams (section headers confirm):

| Lines (approx) | Section | Target module |
|---|---|---|
| 100–520 | token usage, cost, resilience wrapper, `LLMResponseCache`, `get_gemini`/`get_openai`/`get_llm_for_stage` | **Q1 →** `core/llm_runtime.py` |
| 1948–2470 | `_extract_json_payload`, `_compact_json`, schema-aware null/date coercion, `_sanitize_question_analysis_payload` | **Q2 →** `core/llm_payloads.py` |
| 2476–3300 + 3956–4253 | analyzer prompt profile/blocks/render + the shared truncation engine (`_enforce_prompt_budget`, `_section_aware_truncate`, priorities) | **Q3 →** `core/prompt_assembly.py` |
| remainder | the five public entry points (`classify_query_type`, `get_query_focus`, `llm_generate_plan_and_sql`, `llm_analyze_question`, `llm_summarize`, `llm_summarize_structured`) stay in `core/llm.py` | — |

Per phase:
- **Pure move** — no signature, logic, or logging change; `core/llm.py` re-imports everything
  it previously defined (public *and* the private names tests patch).
- **Patch-surface check first** (see governing rule): `_invoke_with_resilience` is patched by
  `tests/security/test_redteam_gate.py`-adjacent code and `test_guardrails.py`; internal
  callers must keep resolving it late through the `core.llm` namespace, or it stays in
  `llm.py` with only the cache/factories moving.
- Mechanical verification: `git diff --stat` shows moves only; import graph stays acyclic
  (`python -c "import core.llm"`); the targeted suite + security suite green per phase.

**Done when:** `core/llm.py` < ~1,800 lines, each new module has one responsibility, zero
test edits were required (proof the shim layer worked).

### Q4 — Split the share-analysis family out of `agent/analyzer.py` (3,285 lines)

**Boundary:** lines ~97–1500 are a coherent share-decomposition family
(`build_share_shift_notes`, `_share_*`, `_pressure_*`, `_resolve_share_target`,
`generate_share_summary`, …) — move to `agent/share_analysis.py`. `analyzer.enrich`
dispatch and the trendline/CAGR/correlation enrichment stay. `analyzer.py` re-exports moved
names (`main.py` re-exports `generate_share_summary`/`build_share_shift_notes` from it;
`BALANCING_SHARE_METADATA` likewise).

**Risk:** Low-Med — same patch-surface rule; `test_main.py`, `test_combined_share_resolution.py`,
`test_guardrails.py` import/patch through `agent.analyzer` and `main`. **Done when:**
`analyzer.py` < ~1,900 lines, suite green with zero test edits.

### Q5 — `main.py` hygiene

**Boundary:** three mechanical cleanups, one commit each:
1. Replace `from config import *` with explicit imports (the F405 list from ruff is the
   exact needed-names inventory — 36 references), then **delete the per-file-ignore** added
   in the CI phase.
2. Replace deprecated `@app.on_event("startup")` with a `lifespan` context manager
   (FastAPI ≥ 0.93 supports it; pinned 0.109.2 does).
3. Unify version strings (`main.py v20.0` header vs `FastAPI(version="18.6")`) into one
   `__version__` constant; drop the stale `# main.py v18.7` comment block.

**Done when:** ruff passes with the ignore removed; suite green.

### Q6 — One spec for the balancing share pivot

**Boundary:** `BALANCING_SHARE_PIVOT_SQL` and `build_trade_share_cte` (both in
`agent/sql_executor.py`) hand-maintain the same entity→`share_*` column mapping twice; the
P2 review noted they must be kept in sync manually, and `test_sql_executor_pivot.py` asserts
shape on both.

**Artifact:** a single `_SHARE_ENTITY_COLUMNS` spec (entity literal → column alias, plus the
three composite sums) from which both the standalone pivot SQL and the CTE text are rendered.
Existing shape tests in `test_sql_executor_pivot.py` are the acceptance gate — they must pass
**unchanged**; additionally assert the generated strings are equivalent to the current
literals before deleting them (golden-string test during the transition).

**Done when:** one spec, both artifacts generated, shape tests unchanged and green.

---

## What this plan deliberately does NOT do

- **No behavior changes** — routing, prompts, thresholds, and answer content are frozen;
  quality work that changes behavior (few-shots, cross-check policy) stays in the §5.3
  pipeline-failure-diagnostics lane.
- **No async migration / performance work** — separate concern, separate plan if wanted.
- **No new services** (per owner constraint) — A5 uses the existing in-process metrics.
- **No big-bang split** — Q1–Q4 are four bounded extractions with shims, not a re-layout.

## Expected grade movement

- **Code quality C+ → B-** after Q1–Q4 (file-size pathology resolved, lexicon duplication
  gone after A3) **→ B** with Q5–Q6 and the already-landed lint/coverage CI gates.
- **Architecture B- → B** after A1 (honest doc), A2 (no orphan code), A4 (transitional code
  has a dated decision), A5 (misrouting measured, not guessed).
