# F10 B4.A — Backend Evidence Ledger — 2026-07-19

Repository-side completion of the B4.A checklist in
[`f10_blocker_remediation_plan_2026-07-18.md`](./f10_blocker_remediation_plan_2026-07-18.md)
against the exact B2/B3 backend candidate. Items that require the deployed
production artifact, the Railway control plane, or provider spend are recorded
as `Manual verification pending` with a runbook, per the independent-repository
rule.

## 1. Candidate identity

| Field | Value |
|---|---|
| Candidate | `refactor/review-phase-fixes` @ `b628f7881175fc12e47da0f87570411d65c0e789` |
| CI | Run [`29678536060`](https://github.com/IrakliGLD/langchain_railway/actions/runs/29678536060) — **every step green**, first-ever pass of the full-test/coverage-floor step |
| Dependency posture | 68-pin hashed lock, `pip-audit` closure gate green, **zero advisories / zero waivers** |
| Later commits | Documentation-only, recorded as such in the B0 ledger §11 — the candidate identity is unchanged |

## 2. Clean-checkout quality, security, and dependency gates

| B4.A requirement | Evidence |
|---|---|
| Complete pytest + Ruff from a clean checkout/container | CI run above: clean ubuntu checkout, Python 3.11, `--require-hashes` lock install, Ruff, full suite **1,710 passed**, coverage floor 83.34% ≥ 80. Local 3.14 diagnostic runs agree (1,710 ×4 on 2026-07-19). True container smoke (Docker build) is emitted by the release-evidence workflow (§7). |
| Security adversarial suite | Green in CI and in the dedicated local evidence run (§4 subset). |
| Formal red-team score gate | CI step green. Fresh local run: **pass_gate true, score 1.0**, block_rate 1.0, false_block_rate 0.0, warn_detect 1.0, grounded_accept 1.0, ungrounded_reject 1.0, hard_failures []. |
| Dependency audit + SBOM review | Audit: CI `pip-audit` gate over the full hashed closure — clean. SBOM: produced by `Backend release evidence` at the candidate SHA — **Manual verification pending** (operator reviews and archives the artifact, B0 §11). |

## 3. HTTP contract, authentication, and typed errors

All in `tests/test_main.py` (+ `tests/test_auth_negative.py`), green in CI and
in the dedicated evidence run:

- **Signed gateway `/ask`**: actor-assertion HMAC binding accepted; unsigned,
  partial, malformed, stale, future, or tampered assertions fail closed (401).
- **Replay denial**: exact-operation replay → 409 (`test_main.py:282`);
  cross-actor session-token replay rejected before the pipeline runs
  (`test_main.py:700`); overlapping session turn → 409 (`test_main.py:764`).
- **Request size**: pre-parse ASGI body-limit middleware → 413
  `REQUEST_TOO_LARGE` (`test_main.py:535`, `:1115`), invalid Content-Length → 400.
- **Typed error envelope** covers 400/401/403/409/413/422/429/500/503
  (`test_main.py:1024`); pipeline HTTP failures stay status-preserving and
  public-safe.
- **Bearer policy** (dormant in production): full PyJWT 2.13 rejection matrix
  pinned in `tests/test_auth_negative.py`.

Live signed `/ask` against the deployed artifact is a §6 smoke item.

## 4. Failure and reliability matrix

Extends [`f4_deadline_failure_matrix_2026-07-16.md`](./f4_deadline_failure_matrix_2026-07-16.md)
(which already maps browser/Edge aborts, backend-before-provider-send,
provider-response-lost, permanent failures, 429 fallback, DB statement
timeout, and secondary-evidence timeout to their tests). Dedicated fresh run
of the full evidence subset: **428 passed** (security suite, F4 semantics, DB
gateway/coordinator, provider runtime/breakers/degradation, evidence planner,
pipeline deadline, application runtime, main, sessions, rate limits, metrics
concurrency, config).

| B4.A scenario | Automated evidence |
|---|---|
| Normal request | `test_main.py` happy-path contract tests (v1 + additive v2). |
| Slow query / statement timeout | Transaction-local `statement_timeout` = min(configured, remaining − cleanup); insufficient cleanup budget prevents checkout (F4 semantics + DB gateway tests). |
| Pool exhaustion | Saturation is bounded without touching the pool (`test_db_gateway.py:207`); checkout timeout releases capacity and feeds the breaker path (`:252`); pool capacity guard rejects configs without application+control slots (`test_config.py:70`). |
| Breaker open | DB and LLM breakers fail fast without touching engine/invoke (`tests/security/test_adversarial_suite.py:148`, `:166`); open breaker rejects before global admission (`test_db_gateway.py:269`) and before claiming/sending a provider attempt (`test_provider_invocation_runtime.py:134`); per-provider breaker independence (`test_provider_breakers.py`). |
| Readiness under saturation | Reserved control lane reaches readiness while the application lane is saturated (`test_db_gateway.py:229`); backpressure gate rejects when saturated (`test_adversarial_suite.py:186`); `/readyz` snapshot contract preserved (`test_main.py:1181`, `test_application_runtime.py:74`). Live p95 probe on the deployed artifact → §6. |
| Request cancellation | Coordinator drain cancels queued work and waits for running work (`test_db_work_coordinator.py:115`); stale singleflight cancellation cannot cancel the replacement owner (`test_llm_cache_singleflight.py:55`); browser/Edge abort semantics are frontend-attested (F4 matrix rows 1–4). |
| Simultaneous primary/secondary evidence load | Prefetch executes steps concurrently with exception isolation (`test_evidence_planner.py:1215`, `:1262`); secondary budgets are capped by remaining request time with context propagation (pipeline deadline tests); no per-request thread pool (`test_db_work_coordinator.py:165`). |
| Backend-before-provider-send / provider-response-lost / DB-timeout / secondary-timeout (deterministic fakes) | F4 matrix rows with `tests/test_f4_deadline_semantics.py` and provider runtime tests; ambiguous delivery is recorded and never retried or failed over. |

## 5. Idempotency and duplicate-charge protection

- Provider attempts are keyed by hashed actor binding + request ID + provider
  + stage; a duplicate completed, in-flight, or ambiguous stage is rejected,
  and failed calls finalize exactly once with breaker updates keyed by
  delivery disposition (`test_provider_invocation_runtime.py:161`).
- The gateway assertion replay cache turns an exact repeated signed operation
  into 409 before any work starts (`test_main.py:282`).
- Cross-process, the durable Supabase chat-operation ledger remains the
  charging authority (frontend-repo evidence; F4 matrix), under the documented
  one-replica constraint.
- Live proof (same signed request ID twice against production → one provider
  attempt, one charge) → §6.

## 6. Production smoke runbook — after B3 promotion (safe set only)

Run against the deployed candidate, in order; record results in B0 ledger §11:

1. `GET /versionz` with `X-App-Key: <ENAI_EVALUATE_SECRET>` → `git_sha` equals
   the candidate byte-for-byte; `/healthz`, `/readyz` green.
2. One real signed chat request through the Edge gateway (happy path), then a
   conversation continuation — answer sane, quota decremented once.
3. Invalid signature and replayed assertion → 401/409, no pipeline execution,
   no charge.
4. Ordinary browser abort mid-request → operation terminal state consistent,
   no orphan retry.
5. Readiness probe under the approved load limit (§8) — no degradation.
6. Duplicate-charge probe: resend one identical signed operation → exactly one
   provider attempt/charge in the ledger.
7. Confirm Railway shows one replica, autoscaling disabled, one worker.

Destructive and provider-ambiguity injection stays in the isolated test
harness — never in production (plan §2.6).

## 7. Manual verification pending (deployed-artifact items)

| Item | Owner | Blocked on |
|---|---|---|
| `Backend release evidence` run at the candidate SHA (SBOM + audit JSON + image digest + Docker smoke) | Irakli | Operator workflow dispatch |
| Railway deployment/rollback IDs, one-replica/autoscaling attestation | Irakli | B3 deploy |
| Production smoke §6 results | Irakli | B3 deploy |
| Production chat load within §8 parameters | Irakli | §8 approved 2026-07-19 → operator run only |

## 8. Production chat-load parameters — APPROVED 2026-07-19 by Irakli

Approver: **Irakli** (B0 §6 waiver/approval authority and rollback owner).
Approved 2026-07-19. The load run itself remains an operator action executed
under this envelope; it must be aborted immediately if any threshold below is
hit.

| Parameter | Approved value |
|---|---|
| Maximum concurrency | 2 |
| Total request count | ≤ 20 |
| Provider spend ceiling | ≤ $2 (abort at ceiling) |
| Test window | 15 minutes, low-traffic hour |
| Abort thresholds | Any duplicate provider attempt/charge; error rate > 10%; `/readyz` degradation; DB saturation; ceiling reached |
| Rollback owner | Irakli (previous Railway deployment ID recorded in B0 §11 before the window) |

The backend production-evidence **governance** is now unblocked: with these
parameters approved and `/versionz` identity verified (B0 ledger), the
remaining §7 rows are pure operator execution (release-evidence artifact
review, Railway ID capture, §6 safe smoke probes, and the bounded load run
within this envelope). No provider spend or production load occurs until the
operator runs it.

## 9. Exit status

**Repository-side B4.A: complete** — every automatable requirement has green,
cited evidence at the exact candidate SHA. The B4 exit gate itself remains
open until the §7 rows are filled from the deployed artifact.
