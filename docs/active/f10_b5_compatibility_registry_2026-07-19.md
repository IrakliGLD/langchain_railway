# F10 B5 — Backend Compatibility Registry — 2026-07-19

Required first step of Phase B5 in
[`f10_blocker_remediation_plan_2026-07-18.md`](./f10_blocker_remediation_plan_2026-07-18.md):
every compatibility candidate classified as **expired**, **rollout**,
**resilience fallback**, or **protocol/data compatibility**, each retained
entry with owner, removal criterion, and deadline. Owner for every entry:
**Irakli** (B0 §6). Deadlines assume B3 promotion lands in July 2026; slide
them with B3, not silently.

## 1. Expired — removed now

| Item | Disposition |
|---|---|
| `ENABLE_AGENT_LOOP` flag, its two pipeline branches, the `agent_loop_blocked_by_policy` trace field, 23 test monkeypatches, and two trace-shape assertions | **Removed in commit `a99e51c`** (this phase). The loop implementation itself was deleted in an earlier phase; this was dead residue. No public schema change (field was trace-only; contract drift gates green). |
| `google.generativeai` lazy fallback branch in `knowledge/vector_embeddings.py` | **Expired-candidate, deferred**: unreachable in the hashed-lock container (`google-genai` is a hard pin, so the ImportError arm can never execute). Removal criterion: fold into the next change that touches the embedding provider class. Deadline: 2026-08-31. Kept this phase to avoid touching provider code in a cleanup slice. |

## 2. Rollout — retained while collecting two-release evidence (B5.A.4)

Activation one-at-a-time via the existing canary runbook
([`p4_f3_canonical_pipeline_activation_runbook_2026-07-16.md`](./p4_f3_canonical_pipeline_activation_runbook_2026-07-16.md));
after two stable production releases with counters/incidents/quality
recorded, make the selected behavior the only path and delete the switch and
holdback code.

| Switch (default) | Removal criterion | Deadline |
|---|---|---|
| `EVIDENCE_FINALIZATION_MODE` (`shadow`) | Two stable releases at `enforce`; finalization counters clean | 2026-08-31 |
| `PLAN_VALIDATION_MODE` (`shadow`) | Two stable releases at `enforce`; no false-reject incidents | 2026-08-31 |
| `ENABLE_HONEST_TERMINAL_OUTCOMES` (`false`) + its `p4_rollout` holdback | Two stable releases on; terminal-outcome mix reviewed | 2026-08-31 |
| `ENABLE_EVIDENCE_REANALYSIS` (`false`) | Two stable releases on; reanalysis quality/deadline metrics reviewed | 2026-08-31 |

Completed default-on rollout switches whose `False` branches are the legacy
path (criterion for each: two stable post-B3 releases at default, then delete
the switch and its `False` branch; deadline 2026-08-31):
`ENABLE_TYPED_TOOLS`, `ENABLE_EVIDENCE_PLANNER`,
`ENABLE_QUESTION_ANALYZER_HINTS`, `ENABLE_SKILL_PROMPTS_SUMMARIZER`,
`ENABLE_SKILL_PROMPTS_PLANNER`, `ENABLE_VECTOR_KNOWLEDGE_HINTS`.

Shadow/diagnostic observers (delete when their comparison purpose completes,
per [`VECTOR_KNOWLEDGE_ROLLOUT.md`](./VECTOR_KNOWLEDGE_ROLLOUT.md) and the
analyzer shadow review): `ENABLE_QUESTION_ANALYZER_SHADOW`,
`ENABLE_VECTOR_KNOWLEDGE_SHADOW`, `ENABLE_CONTRACT_CONTINUITY`. Deadline
2026-09-30.

## 3. Resilience fallback — intentionally retained (B5.A.5)

| Item | Rationale / removal criterion |
|---|---|
| Legacy SQL fallback path (planner `generate_plan` → `sql_executor`) | Intentional failure behavior behind typed tools. Remove only when production counters, correctness evaluation, and a safer terminal behavior prove it unnecessary — not because the name says legacy. |
| `legacy_text_fallback` summary path in the summarizer | Same rule; it is the safe degradation for structured-summary failures, with claims derived for the provenance gate. |
| Application-owned OpenAI failover after proven pre-delivery rejection (`safe_to_fallback`) | Permanent resilience design (F4); not a compatibility path. |

## 4. Protocol/data compatibility — remove only after inventory proves zero use

| Item | Removal criterion | Deadline |
|---|---|---|
| `GATEWAY_ACTOR_ASSERTION_MODE` `optional` value (B5.A.2) | Production traffic logs at the promoted release prove 100% of gateway requests carry valid assertions over the observation window; then make assertion unconditionally required and retire the env knob entirely (not just the default). Signature freshness/replay tests are retained. | First maintenance window after B3 + log review; target 2026-08-15 |
| Legacy secret-name fallbacks `GATEWAY_SHARED_SECRET`/`SESSION_SIGNING_SECRET`/`EVALUATE_ADMIN_SECRET` (`config.py` startup validation) (B5.A.3) | Railway environment inventory confirms canonical `ENAI_*` names present, and one rollback deployment is rehearsed on a release that still accepts both; then delete the fallback reads. | 2026-08-15 |
| `PROMPT_BUDGET_MAX_CHARS` legacy single-knob mapping | Deployment env inventory shows no reliance on the single knob (per-stage budgets or defaults in use); then delete the mapping. | 2026-08-31 |
| `ENAI_AUTH_MODE=gateway_and_bearer` / dormant `ENABLE_PUBLIC_BEARER_AUTH` surface | Product decision to never ship public bearer auth. Until then it is a guarded, default-off protocol option with its negative-test matrix (`tests/test_auth_negative.py`); production stays `gateway_only`. | Decision by B6 closure |
| `Question.user_id` legacy field (public v1 contract) | Part of the published v1 API; removable only through the versioned-contract process together with the frontend repository's chat-gateway v1 retirement (B5.B item 1). | Tracks B5.B v1 retirement |

## 5. Explicitly out of scope (not compatibility paths)

Operational gates and product configuration, excluded from removal planning:
`ENABLE_METRICS_ENDPOINT`, `ENABLE_EVALUATE_ENDPOINT`,
`ENABLE_TRACE_DEBUG_ARTIFACTS`, `ENAI_FIXTURE_CAPTURE_MODE`, `PIPELINE_MODE`.

## 6. Exit-gate position

- Expired paths: deleted (§1), except one deferred expired-candidate with a
  named criterion and deadline.
- Every retained path: classified with owner, criterion, deadline (§2–§4).
- Two-release candidates: evidence collection starts at B3 promotion; none may
  be forced into the first release (plan §3).
- Rollback remains possible through attested releases; no dormant unsafe code
  path was reintroduced.
