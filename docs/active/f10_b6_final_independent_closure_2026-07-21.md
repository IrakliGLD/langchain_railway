# F10 B6 — Final Independent Closure Re-Audit — 2026-07-21

## Decision

**F10 is closed at the final post-merge identity.** No unwaived High or Critical
finding remains. The former pre-merge closure note is historical; PR #137 merged
`docs/f10-b6-f3-evidence` to `main`, and Railway deployed that merge.

## Final identities

| Surface | Exact identity | Independent result |
|---|---|---|
| Frontend/browser + all nine Edge functions | `fc44fd40946bb0772ab4f178ac376196bec21498` | Release evidence, authenticated smoke, Live Browser Proof and axe evidence reviewed green |
| Backend source / protected `/versionz` | `0684dc172eb2bb10a17a2e80941a6940b0882f2d` | PR #137 merge and `/versionz` evidence agree |
| Backend release evidence | run `29812847819`, job `88577462455`, artifact `8488198995`, artifact digest `sha256:ab884b891de35f688ceef49644cde6190de2705c00c5f512acf4336a4ebfbd12` | Exact checkout, embedded/OCI identity, SBOM, pip-audit, non-root and lock gates green |
| Railway active deployment | `e2e73849-c47d-4f4f-8073-76edd2e0df95` | Successful, one replica, `/healthz` and `/readyz` healthy |
| Railway runtime image | manifest `sha256:319f6774f8197acd88941abae1b81a57bb10d19d98fa28a82e9d5b63c3b5336b`; config `sha256:75cca0a003a757719a3dd9389d7015f7471877178c6a0c2c7f653317e7e0e396` | Captured from Railway Build Logs and bound to the final SHA |
| Rollback target | `feff19e0-69d6-42e9-818a-757d76090e2e` (`65cf93b`; prior manifest `sha256:b3791219…`) | Rollback/roll-forward rehearsal retained and cross-referenced |
| Runtime dependency | `langsmith==0.10.7` | Confirmed in the active Railway console |

The binding is recorded in the [F3 evidence runbook](./f10_b6_f3_evidence_runbook_2026-07-20.md)
§8 and the [F2 freeze record](./f10_b6_f2_freeze_evidence_2026-07-20.md) final-identity appendix.

## B6 closure matrix

| Gate | Result | Evidence / disposition |
|---|---|---|
| Backend quality and security | **Pass** | Local independent full suite: **1,710 passed** with `--basetemp=.tmp_pytest_b6_2`; focused evidence suite 74 passed; provider/deadline regression **23 passed** after correcting Gemini's seconds-vs-milliseconds timeout boundary; Ruff clean; lock check 68 pins current; release SBOM/pip-audit green; red-team gate green |
| Frontend, Edge and release identity | **Pass** | Exact-SHA release artifact, Edge all-nine self-verification, authenticated smoke, Live Browser Proof and three axe scans reviewed green at `fc44fd4` |
| Disposable DB regression | **Pass by immutable carry-forward proof** | Frontend CI run `29701996275` / #271, job `88232468171`, throwaway Supabase `ufosbrdhjrkaagjaltno`; DB regression green and `database/baseline` + `database/tests` trees byte-identical from `a4c24a2` to `fc44fd4` |
| Accessibility | **Pass with approved scoped disposition** | Structural/authenticated checklist and axe green; actual SR listening is explicitly governed by `F10-B6-A11Y-01`, approved 2026-07-21, not misrepresented as performed evidence |
| Representative chat load | **Pass by approved expiring waiver** | `F10-B6-LOAD-01`, approved 2026-07-21, expires 2026-08-20; covers only the load-observation row and requires the envelope before expiry/renewal |
| Runtime immutable identity (REL-02) | **Pass** | Final Railway manifest digest is bound to final source SHA, `/versionz`, release artifact/SBOM/audit evidence and rollback target |
| Compatibility and architecture reconciliation | **Pass** | Expired Gemini fallback removed; bearer surface explicitly retained as default-off protocol compatibility with owner, criterion and review date; loop and registry/audit stale claims reconciled |

## A–F assessment

| Perspective | Grade | Rationale |
|---|---:|---|
| Functional correctness and query pipeline | **A** | Full backend regression and release/browser evidence are green at the final identities |
| Architecture and maintainability | **A-** | Compatibility decisions and registry/architecture records are reconciled; large orchestration modules and planned follow-on cleanups remain intentionally open |
| Security and privacy | **A-** | Exact lock, SBOM, pip-audit, auth-negative and red-team gates are green; dormant bearer compatibility remains guarded and default-off |
| Reliability, concurrency and error handling | **A-** | Deterministic barrier concurrency proof, bounded admission, session locking, provider breakers, cancellation and rollback evidence are green |
| Performance and scalability | **B+** | One replica and an eight-request in-flight ceiling are deliberate; representative chat load is waived, so no measured throughput claim is made |
| Frontend UX and accessibility | **A-** | Live browser/axe/structural checks are green; SR listening remains an explicitly approved, dated disposition |

**Overall: A-.** This is closure-grade A-range quality with two consciously bounded
operational limitations: load is an expiring evidence waiver, and real screen-reader
listening is a documented scope disposition. Neither is an unowned or hidden release
blocker.

## Residual follow-up (not closure blockers)

1. Run the approved representative chat-load envelope, or renew `F10-B6-LOAD-01`,
   before 2026-08-20.
2. If the product gains an assistive-technology user base, perform and attach the
   NVDA/VoiceOver listening attestation for `fc44fd4`.
3. Revisit dormant bearer compatibility by 2026-09-30 under the recorded removal
   criterion; production remains `gateway_only`.
