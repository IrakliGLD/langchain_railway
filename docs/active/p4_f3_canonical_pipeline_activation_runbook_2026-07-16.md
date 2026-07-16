# F3 canonical pipeline activation runbook

**Date:** 2026-07-16
**Scope:** P4 canonical evidence finalization, plan enforcement, honest terminal outcomes, frontend outcome rendering, and evidence-triggered re-analysis
**Repositories remain independent:** backend `D:/Enaiapp/langchain_railway`; frontend `D:/export_enai/_repo_sync`

## 1. What this release changes

F3 adds request-scoped deterministic canaries around the four behavior-changing P4 gates. It does not enable any gate by default:

- evidence finalization remains `shadow`;
- plan validation remains `warn`;
- honest terminal outcomes remain off;
- evidence re-analysis remains off.

The frontend adds distinct visible states for `evidence_unavailable`, `clarification_required`, `policy_blocked`, and `transient_failure`. It consumes the already-versioned `chat-gateway-v2` / `chat-edge-v3` contract; no database patch or Edge Function source change is introduced by F3.

## 2. Rollout controls

| Behavior | Master control | Percentage control | Holdback behavior | Immediate rollback |
|---|---|---|---|---|
| Canonical evidence finalization | `ENAI_EVIDENCE_FINALIZATION_MODE=enforce` | `ENAI_EVIDENCE_FINALIZATION_ENFORCE_PERCENT` | `shadow` | set mode to `shadow` |
| Plan rejection before execution | `ENAI_PLAN_VALIDATION_MODE=enforce` | `ENAI_PLAN_VALIDATION_ENFORCE_PERCENT` | `warn` | set mode to `warn` |
| Honest evidence-unavailable answer | `ENAI_ENABLE_HONEST_TERMINAL_OUTCOMES=true` | `ENAI_HONEST_TERMINAL_OUTCOMES_PERCENT` | legacy conceptual route plus shadow counter | set master to `false` |
| One evidence-triggered re-analysis | `ENABLE_EVIDENCE_REANALYSIS=true` | `ENAI_EVIDENCE_REANALYSIS_PERCENT` | detection/counters only | set master to `false` |

Each percentage accepts an integer from 0 through 100 and defaults to 100. The 100 default preserves the historical meaning of explicitly enabling a master control. Because all master controls keep their safe production defaults, deploying the code without adding these percentage variables changes no request behavior.

Cohorts are deterministic per gate and prefer the gateway-verified actor ID, then the signed server session ID, then the request ID. Raw identifiers and hashes are not stored in metrics. Partial rollout without any stable identifier is `ineligible` and fails closed into the holdback behavior.

## 3. Preconditions

Before any behavior-changing canary:

1. Deploy the backend F3 implementation with the current safe master-control values.
2. Keep Railway at one worker and one replica; `/metrics` is process-local.
3. Keep production in gateway-only mode with verified actor assertions so partial cohorts are stable.
4. If `/metrics` is enabled, keep it admin-authenticated and network-restricted.
5. Deploy the frontend F3 browser artifact before enabling honest terminal outcomes. No F3 Edge Function or SQL deployment is required.
6. Confirm the frontend and Edge still negotiate `chat-edge-v3` / `chat-gateway-v2`.
7. Preserve a one-change rollback deployment for each gate. Do not enable multiple P4 master controls in one release.

## 4. Baseline observation

Run the deployed backend with:

```text
ENAI_EVIDENCE_FINALIZATION_MODE=shadow
ENAI_PLAN_VALIDATION_MODE=warn
ENAI_ENABLE_HONEST_TERMINAL_OUTCOMES=false
ENABLE_EVIDENCE_REANALYSIS=false
```

Record the beginning and ending `/metrics` snapshots and review representative protected traces. At minimum capture:

- `evidence_finalization_events`: shadow frames, empty/no-tool/no-adapter shares, and validation-gap trace details;
- `plan_validation_events`: reject/warn rules and the plans that would have been prevented;
- `chart_source_events`: canonical aligned/filtered paths, raw fallbacks, and especially `raw_ctx_df_frame_unmatched`;
- `terminal_outcome_events`: normal outcomes and `evidence_unavailable_shadow`;
- `evidence_anomaly_events`: `primary_empty` and `period_gap`;
- `p4_rollout_events`: all gates should be `disabled` while master controls are safe;
- sampled `chat-gateway-v2` responses: answer provenance and every chart identity must agree on period, filter, unit, and provenance references.

Do not advance if the baseline contains unexplained frame gaps, chart/frame mismatches, malformed contracts, or high anomaly rates. Fix those root causes while gates remain safe.

## 5. Activation sequence

Advance only one behavior at a time. For every percentage step, deploy a new Railway configuration, run the same golden/smoke query set, capture before/after metrics, and observe a normal traffic window. Time alone is not an advance criterion.

### 5.1 Canonical evidence finalization

1. Set `ENAI_EVIDENCE_FINALIZATION_MODE=enforce`.
2. Start with `ENAI_EVIDENCE_FINALIZATION_ENFORCE_PERCENT=0`; verify `evidence_finalization:holdback`.
3. Advance to 5, then 25, then 100.
4. At each step confirm:
   - no increase in request errors, timeouts, grounding failures, or empty answers;
   - canonical finalization produces the expected frame types;
   - deterministic shapes do not add narrative model calls;
   - `raw_ctx_df_frame_unmatched` is absent or individually justified;
   - answer/chart period, filter, unit, and provenance identities match.

Rollback: set `ENAI_EVIDENCE_FINALIZATION_MODE=shadow`.

### 5.2 Plan validation enforcement

Begin only after evidence finalization is stable at the approved percentage.

1. Set `ENAI_PLAN_VALIDATION_MODE=enforce`.
2. Set `ENAI_PLAN_VALIDATION_ENFORCE_PERCENT=0`, then 5, 25, and 100.
3. For every enforced reject, verify from traces that no tool or database call occurred.
4. Review clarification quality and reject rules; a false reject is a rollback condition.

Rollback: set `ENAI_PLAN_VALIDATION_MODE=warn`.

### 5.3 Honest terminal outcomes

Begin only after the F3 frontend browser artifact is deployed and history/live outcome smoke passes.

1. Set `ENAI_ENABLE_HONEST_TERMINAL_OUTCOMES=true`.
2. Set `ENAI_HONEST_TERMINAL_OUTCOMES_PERCENT=0`, then 5, 25, and 100.
3. Confirm data-primary SQL failures return `evidence_unavailable`, contain no numeric claims, remain HTTP-success anti-retry-storm outcomes, persist correctly, and render as “Data unavailable” in live and restored history.
4. Smoke `clarification_required`, `policy_blocked`, and `transient_failure` presentation separately.

Rollback: set `ENAI_ENABLE_HONEST_TERMINAL_OUTCOMES=false`.

### 5.4 Evidence-triggered re-analysis

Activate last. First collect at least two weeks of `evidence_anomaly_events`, investigate high anomaly rates as routing defects, and run the routing golden set with re-analysis enabled.

1. Set `ENABLE_EVIDENCE_REANALYSIS=true`.
2. Set `ENAI_EVIDENCE_REANALYSIS_PERCENT=0`, then 5, 25, and 100.
3. Confirm every retry has one anomaly reason, runs at most once, respects the request deadline, and leaves no stale response mode, plan, evidence, frame, metrics, or terminal outcome.
4. Compare extra provider/database cost and latency against the anomaly rate. Re-analysis is a rare safety net, not a normal route.

Rollback: set `ENABLE_EVIDENCE_REANALYSIS=false`.

## 6. Advance and rollback criteria

Do not advance a gate when any of the following occurs:

- `p4_rollout_events` reports `ineligible` traffic in a gateway-authenticated partial rollout;
- invalid plans reach a tool/database call;
- answer and chart identities disagree;
- evidence-unavailable responses contain numeric claims or render as ordinary success;
- re-analysis occurs twice, crosses the request budget, or retains stale state;
- request error/timeout/grounding-failure rates materially exceed the recorded baseline;
- the deployed contract version or artifact SHA is not the intended release.

Rollback the affected master control first. Preserve metrics/traces, record the deployment SHA and time window, identify the root cause, add a regression, and repeat from 0 percent.

## 7. Evidence record

For each deployment record:

- backend commit SHA and Railway deployment ID;
- frontend commit SHA for the browser deployment;
- exact master-control and percentage values;
- observation start/end and request count;
- before/after snapshots of the six F3 metric groups;
- golden/smoke query results and sampled gateway identities;
- error, timeout, grounding, raw-fallback, reject, degraded-outcome, and anomaly rates;
- decision: advance, hold, or rollback;
- reviewer and rollback evidence.

F3 is production-complete only after all four gates reach their approved deployed state and the P4 exit evidence is attached. Local implementation and tests alone do not close that operational gate.
