# Medium-Issue Fix Plan

**Date:** 2026-06-10
**Source:** [`audit_2026-06-10.md`](audit_2026-06-10.md)
**Workflow:** Each phase follows [`developer-phased-audit`](../../skills/developer-phased-audit/SKILL.md):
state phase boundary → produce one artifact → mechanically verify → **targeted suite green**
(`python -m pytest tests/ --ignore=tests/security -q`) → independent audit → fix findings →
re-plan. Phases that touch guardrails (P1, P3) also run the security gate
(`python -m pytest tests/security -q` + `python -m guardrails.redteam_gate`).

## Implementation status (updated 2026-06-10)

| Phase | Status | Verification |
|---|---|---|
| P1 — reject stacked SQL | ✅ Implemented | Full targeted suite 1052 passed; 6 new tests in `test_main.py::TestSQLValidation` |
| P2 — supply-filter on explicit intent only | ✅ Implemented | Included in 1052; 8 tests in `test_sql_executor_side_filter.py` |
| P3 — multilingual firewall + bearer history firewalling | ✅ Implemented | Security suite 19 passed; redteam gate score 1.0 (block 1.0 / false-block 0.0); 6 history tests + multilingual adversarial cases |
| P4 — least-privilege DB role | ⚠️ Partial — repo artifacts only | `scripts/least_privilege_api_role.sql` + drift test `test_config.py::test_readonly_role_grants_match_whitelist` (passes). **Live privilege boundary must be applied in Supabase + `SUPABASE_DB_URL` repointed — not verifiable in sandbox.** No app-code change. |
| P5 — Redis shared store | ⏭️ Skipped (owner decision) | Not implemented; Redis dependency declined. Multi-replica rate-limit/session correctness (S3) and stale-key eviction (S4) remain open as documented recommendations. |

Notes: the full targeted suite is ~16 min run alone (the earlier 2.5h was concurrency
contention from two simultaneous runs). P4 is a Supabase access-rule change applied by
repointing `SUPABASE_DB_URL` at the new read-only role — no pipeline/query code changes.

---

## Scope

Medium-severity items from the audit, plus S6 (Low/Med) folded into P3 since it shares the
firewall touch-point.

| Phase | Item | Audit ref | Effort | Risk | Depends on |
|---|---|---|---|---|---|
| P1 | Reject stacked / multi-statement SQL | S1 | S | Low | — |
| P2 | Stop silently dropping rows in supply-side filter | L1 | S | Low | — |
| P3 | Multilingual firewall + history firewalling | S2, S6 | M | Low | — |
| P4 | Least-privilege DB role | S7 | M | Med | ops/DB migration |
| P5 | Shared store for sessions + rate limits | S3 | L | Med | infra (Redis) |

Ordering rationale: P1/P2 are contained, high-confidence code changes with no infra
dependency — land them first. P3 is additive (new patterns, no removal of existing rules).
P4 and P5 require infra coordination (a DB role, a Redis instance), so they trail and each
ships behind a flag/env so rollout is reversible. **P1 and P4 are complementary** — P4
neutralizes most of S1 at the database layer; P1 keeps the app-layer guard honest. Do both.

---

## P1 — Reject stacked / multi-statement SQL  (S1)

**Boundary:** Make the SQL safety layer reject any input that parses to more than one
statement, so validation can never inspect statement #1 while statement #2 reaches the driver.

**Artifact:** A multi-statement guard in
[`core/sql_generator.py`](../../core/sql_generator.py).

**Approach (deterministic, no LLM):**
- In `sanitize_sql` (currently uses `sqlglot.parse_one`), additionally parse with
  `sqlglot.parse(sql, read="postgres")` and raise `HTTPException(400, "Only a single
  read-only statement is allowed.")` when the list length > 1 (after dropping trailing
  empty/`None` parses from a stray `;`).
- Keep the existing single-root + forbidden-node checks.
- Confirm the **re-rendered** AST is not required — guarding count is sufficient because the
  raw string is what executes; a count guard plus the existing `parse_one` root check covers
  it. (Do *not* switch to executing `parsed.sql()` in this phase — re-rendering risks dialect
  drift in the typed-tool-adjacent SQL; out of scope.)
- `simple_table_whitelist_check` stays as-is; it now only ever sees single-statement input.

**Mechanical verification:** `sanitize_sql("SELECT 1; SELECT 2")` raises 400;
`sanitize_sql("SELECT 1;")` (trailing semicolon only) still passes.

**New tests** (`tests/test_query_validation.py` or a new `tests/test_sql_sanitize.py`):
- positive: single SELECT, single SELECT with trailing `;`, CTE, UNION still pass.
- negative: `SELECT … ; SELECT … from auth.users`, `SELECT … ; DROP …`, comment-smuggled
  `SELECT … ;-- \n SELECT …` all raise 400.

**Targeted suite + security gate must be green.**

**Risk:** A legitimate LLM-emitted query containing an intra-string `;` could trip sqlglot's
splitter — verify sqlglot treats `;` inside a string literal as data, not a separator (it
does), and include a test for `WHERE note = 'a;b'`.

**Done when:** stacked-statement inputs are rejected with 400 and all suites green.

---

## P2 — Supply-side filter no longer drops rows on total queries  (L1)

**Boundary:** On the legacy SQL path, only narrow `type_tech` rows when the query clearly
targets one side of the market. Stop defaulting ambiguous/total queries to supply-only.

**Artifact:** Revised `type_tech` narrowing block in
[`agent/sql_executor.py`](../../agent/sql_executor.py) (the `if "type_tech" in df.columns:`
branch, ~line 343).

**Approach (deterministic):**
- Keep the explicit branches: demand/loss/export → DEMAND side; "transit" → TRANSIT side.
- Replace the unconditional `else: supply_df` default. Apply the **supply** filter only when
  the query has an explicit supply/generation signal (e.g. "generation", "supply",
  "გენერაცია", "выработка"). When neither side is signalled (a total/ambiguous query), keep
  **all** rows.
- When a side filter *is* applied, record it on `ctx` (e.g. a `tool_match_reason` note or a
  trace field) so the summarizer can disclose "supply side only" rather than implying a total.

**Why general, not per-symptom:** the fix is "filter only on explicit intent" — it removes a
silent default rather than special-casing one failing question (per
[`feedback_prefer_general_solutions`]).

**New tests** (extend `tests/test_sql_executor_pivot.py` or new
`tests/test_sql_executor_side_filter.py`, mocking `execute_sql_safely` to return a df with
both supply and demand `type_tech` rows):
- "total electricity in 2024" → all rows retained.
- "generation in 2024" → supply rows only.
- "electricity consumption" → demand rows only.
- "transit 2024" → transit rows only.

**Targeted suite must be green** (watch `test_pipeline_*`, `test_combined_share_resolution`).

**Risk:** Some existing tests may assume the supply default — audit any that break and decide
whether they encoded the bug. Re-plan if a pipeline contract test depends on the old default.

**Done when:** total queries retain all sides; single-side queries still narrow; suite green.

---

## P3 — Multilingual firewall coverage + history firewalling  (S2, S6)

**Boundary:** Extend the Stage-0 firewall block rules to Georgian and Russian, and apply the
firewall's length/content checks to caller-supplied `conversation_history` in public-bearer
mode. Purely additive — no existing English rule is removed or weakened.

**Artifact:** New language-variant patterns in
[`guardrails/firewall.py`](../../guardrails/firewall.py) and a history-inspection call in
[`main.py`](../../main.py) (the bearer-mode history-seed branch, ~line 918).

**Approach:**
1. **Firewall patterns** — add Georgian/Russian variants for `instruction_override`,
   `prompt_exfiltration`, `role_hijack`. Source the phrasings from a native/representative set
   (do not machine-translate blindly — verify with the existing Georgian/Russian strings
   already in `utils/query_validation.py` and `pipeline.py` `_EXPLANATION_ROUTING_SIGNALS` for
   register). Keep them in the same `_BLOCK_RULES` structure so scoring is unchanged.
2. **History firewalling (S6)** — in bearer mode, run each seeded
   `conversation_history` item's `question`/`answer` through `inspect_query` (or at least the
   length cap + control-char scrub) before it enters `bound_history`; drop or sanitize blocked
   items. Gateway-mode history (loaded server-side from the DB by the edge function) is
   trusted and unchanged.

**Per the workflow's LLM-task rules:** this changes a guardrail, so review **disagreement
cases** (queries that should now block in Georgian/Russian but previously passed) explicitly,
not just the new pass cases.

**New tests:**
- `tests/security/test_adversarial_suite.py` (or `test_redteam_gate.py`): add Georgian and
  Russian block cases mirroring the English `BLOCK_CASES`; assert `action == "block"`.
- Add matching cases to `guardrails/redteam_gate.py` `BLOCK_CASES` so the gate score reflects
  multilingual coverage. Confirm `false_block_rate` on the (English + new-language) ALLOW
  cases stays ≤ 0.02 — add benign Georgian/Russian analytical queries to the ALLOW set.
- `tests/test_main.py`: a bearer-mode request whose `conversation_history` carries an
  instruction-override string is sanitized/dropped before reaching the pipeline.

**Both targeted suite and security gate must be green**, and
`python -m guardrails.redteam_gate --min-score 0.92` must still pass.

**Risk:** Over-broad non-English patterns cause false blocks on legitimate Georgian/Russian
analytics. Mitigate with explicit ALLOW cases in the gate and by keeping patterns as specific
phrase sequences (matching the English rules' specificity), not single keywords.

**Done when:** Georgian/Russian jailbreak phrasings block, benign multilingual queries pass,
bearer-mode history is firewalled, gate score ≥ 0.92.

---

## P4 — Least-privilege DB role  (S7)  ★ highest leverage

**Boundary:** The application connects as a Postgres role that can only `SELECT` the nine
whitelisted views/matviews. This is mostly a **DB/ops change**; the code change is small.

**Artifact:** (a) a DB migration creating a read-only role with `GRANT SELECT` limited to the
whitelisted relations and `REVOKE` of everything else; (b) deployment using that role's
connection string in `SUPABASE_DB_URL`; (c) a startup self-check.

**Approach:**
1. **Migration** (lives with the front-end DB migrations under `D:\export_enai\database`, or a
   new `scripts/` SQL file here — coordinate with whoever owns schema): create role
   `enai_readonly`, `GRANT CONNECT`, `GRANT USAGE ON SCHEMA public`, `GRANT SELECT` on exactly
   the `STATIC_ALLOWED_TABLES` set from [`config.py`](../../config.py), and ensure **no**
   access to `auth.*` or other schemas. Vector retrieval also uses `ENGINE` against the
   `knowledge` schema (`VECTOR_KNOWLEDGE_SCHEMA`) — the role must additionally have the
   SELECTs the vector store needs; enumerate those before locking down or vector retrieval
   breaks.
2. **Code:** add an optional startup probe (extend `refresh_schema_map`/`on_startup` in
   `main.py`) that logs a warning if the connection can write (e.g. attempts a harmless
   `CREATE TEMP` in a rolled-back txn and expects failure). No behavior change beyond logging.
3. **Docs:** state in `config.py`/README that `SUPABASE_DB_URL` must be the least-privilege
   role in staging/production.

**Mechanical verification:** with the new role, `SELECT * FROM auth.users` and any write fail
at the DB even if app validation is bypassed; the existing typed tools and vector retrieval
still return data.

**Tests:** unit-testable surface is thin (the privilege boundary is in Postgres). Add a
`test_config.py` assertion that the whitelist the migration grants against matches
`STATIC_ALLOWED_TABLES` (keep them from drifting), and an integration check in the
manual/nightly tier (not `tests/`, per the suite rule against real-DB tests).

**Risk (Med):** under-granting breaks vector retrieval or a matview refresh; over-granting
defeats the purpose. **Enumerate every relation `ENGINE` touches** (price/tariff/tech/trade
views + `knowledge` schema tables + `pg_matviews` reflection query in `refresh_schema_map`,
which reads `pg_catalog` — confirm the role can run it or guard it) before cutover. Roll out
in staging first.

**Done when:** app runs fully on the read-only role in staging; writes and non-whitelisted
reads fail at the DB; targeted suite green (no code-path regression).

---

## P5 — Shared store for sessions + rate limits  (S3)

**Boundary:** Move per-process state (`utils/session_memory._SESSION_STORE` and the three
rate-limit bucket dicts in `main.py`) to a shared backend so limits and sessions are correct
across multiple workers/replicas. Also fixes S4 (stale-key accumulation) for free via TTLs.

**Artifact:** A storage abstraction with two implementations (in-memory default, Redis when
configured), wired into session memory and the sliding-window limiter.

**Approach (additive, flagged):**
1. Add `redis` to `requirements.txt` (none present today) and a `REDIS_URL` config var
   (optional; absent → current in-memory behavior, so single-worker deploys are unaffected).
2. **Session store:** behind an interface, back `_SESSION_STORE` with Redis hashes keyed by
   `session_id`, `EXPIRE`-d to `SESSION_IDLE_TTL_SECONDS`. Signed-token semantics in
   `resolve_session_token`/`issue_session_token` are unchanged — only storage moves.
3. **Rate limits:** replace `_check_sliding_window_rate_limit`'s dict with a Redis sorted-set
   sliding window (`ZADD`/`ZREMRANGEBYSCORE`/`ZCARD`) keyed per subject, `EXPIRE`-d to the
   window. This also evicts stale keys automatically (closes S4).
4. Keep the in-memory path as the fallback when `REDIS_URL` is unset, and when Redis is
   unreachable fail **open or closed** by explicit policy (recommend: limiter fails *closed*
   to a conservative local cap; session store fails to a fresh session) — decide and document.

**Per workflow:** prefer shadow/rollout safety — ship the abstraction with the in-memory impl
first (no behavior change, refactor only, suite stays green), then enable Redis in staging,
then production. Two sub-phases:
- **P5a:** introduce the interface + in-memory impl, migrate call sites, suite green (pure
  refactor; `test_session_memory.py` must still pass unchanged).
- **P5b:** add the Redis impl + `REDIS_URL` wiring + failure policy + tests.

**New tests:**
- `test_session_memory.py`: parametrize over both backends (Redis via a fake/`fakeredis` so
  `tests/` stays fast and external-service-free per the suite rule).
- New `test_rate_limit.py`: sliding-window correctness (N allowed, N+1 blocked, window
  expiry) against both backends.

**Targeted suite must be green at P5a and P5b.**

**Risk (Med):** Redis becomes a new dependency in the request hot path; the failure policy
(step 4) is the critical decision — an unbounded fail-open defeats the limiter, a hard
fail-closed creates an availability dependency. Pin the policy before P5b and cover it with a
test that simulates Redis-down.

**Done when:** with `REDIS_URL` set, two app instances share one rate-limit budget and one
session view; with it unset, behavior is byte-for-byte the current in-memory path; suites green.

---

## Cross-cutting notes

- **Sequencing:** P1 → P2 → P3 can proceed in parallel branches (no shared files). P4 and P5
  need infra tickets opened now (DB role; Redis instance) because their lead time, not their
  code, is the long pole.
- **Each phase is independently shippable** and behind a test gate; none requires a big-bang
  cutover. P4 and P5 are additionally reversible via env (`SUPABASE_DB_URL` role swap,
  `REDIS_URL` unset).
- **Out of scope (Low items):** S5 (proxy-IP extraction), S8 (gateway error-detail leakage),
  S9 (referer check), L2–L7 — track separately; several are one-liners that can ride along
  with the phase that touches the same file (e.g. S8 with P3 since both touch `main.py`).
- **Verification discipline:** "green on the modules I changed" is explicitly *not* sufficient
  per [`targeted-suite.md`](../../skills/developer-phased-audit/references/targeted-suite.md) —
  run the full `tests/ --ignore=tests/security` sweep before each phase's audit step.
