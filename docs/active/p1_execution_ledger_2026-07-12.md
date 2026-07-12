# P1 Execution Ledger

**Date:** 2026-07-12
**Scope:** Independent backend and frontend/Supabase repository tracks
**Release state:** P1.A, P1.1, and P1.2 are committed. P1.3 is not complete or committed because its real Deno lock and generated immutable source manifest do not yet exist.

## Repository identities

| Track | Repository/worktree | Branch | Evidence |
|---|---|---|---|
| P1.A backend | `D:/Enaiapp/langchain_railway` | `refactor/review-phase-fixes` | `5af3ff5` — `Publish P1 chat gateway contract` |
| P1.B frontend | `D:/Enaiapp/p0_frontend_commit_repo` (canonical Git checkout for the application linked to `D:/export_enai`) | `fix/p0-remediation` | `9f0bd6d` — canonical source; `05c0378` — deterministic auth |

The dirty/private legacy copies under `D:/export_enai` were not modified. Their retirement remains an operator action after the canonical deployment is proven.

## Package status

| Package | State | Verified evidence | Remaining action |
|---|---|---|---|
| P1.A gateway contract | Committed, locally verified | Versioned JSON/document contract, request-version mismatch handling, safe request correlation, full backend suite: 1,358 passed | Deploy `5af3ff5` or a descendant before the compatible chat edge function; record deployed backend SHA |
| P1.1 canonical source | Committed, locally verified | 132-difference SHA-256 reconciliation manifest; runtime/build/lint/CI/deploy paths reject the mirror; frontend tests/build passed | Point hosting at the canonical Git repository/commit, then archive private legacy copies |
| P1.2 deterministic auth | Committed, locally verified | Delayed `INITIAL_SESSION`, rapid sign-in/refresh/sign-out, stale profile/admin, timeout/offline, paused-user, protected-route, and cross-tab-style event tests pass | Run real two-tab and offline browser smoke tests after staging deployment |
| P1.3 deployable edge sources | In progress; intentionally uncommitted | Nine functions moved to standard sources; shared auth/CORS/log/error controls; exact Supabase dependency; strict TypeScript shim check; 332 frontend/contract tests, lint, build, YAML parse, and diff check pass | Install pinned Deno 2.1.4, generate `supabase/deno.lock`, generate source manifest, run Deno format/lint/check/test, independently audit the diff, then commit |

## P1.3 fail-closed boundary

The local environment has no Deno or Docker runtime. An attempt to obtain the pinned Deno runtime was rejected by the execution environment's usage-limit gate. No lockfile was fabricated from npm metadata and no placeholder source hash is accepted as evidence.

The frontend worktree contains tooling that intentionally fails while `supabase/deno.lock` is absent. Complete it from `D:/Enaiapp/p0_frontend_commit_repo` with:

```powershell
npm ci
npm run edge:lock
npm run edge:manifest:write
npm run edge:verify
npm test
npm run lint
npm run build
git diff --check
```

The detailed environment, secrets, deploy, proof, and rollback procedure is `docs/active/p1_edge_delivery_runbook.md` in the frontend repository.

## Activation order

1. Complete and commit P1.3's real lock and source manifest.
2. Push the independent backend and frontend commits.
3. Deploy backend commit `5af3ff5` or a tested descendant and verify the `chat-gateway-v1` response header/correlation behavior.
4. Configure protected GitHub staging/production environments and Supabase runtime secrets.
5. Deploy the frontend edge artifact by full 40-character commit SHA through the new workflow.
6. Prove the deployed `X-Enai-Edge-Source` identity, then run active/inactive/admin/authenticated staging smoke tests.
7. Deploy the frontend application from the same canonical repository and record its commit.
8. Promote the same edge SHA to production, retain evidence, and rehearse rollback to the previous approved SHA.

P3.B must not begin deployment until P1.3 is committed and its staging deployment identity is proven. P2.A may proceed independently because it does not depend on the edge delivery cutover.
