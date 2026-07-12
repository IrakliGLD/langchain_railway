# P1 Execution Ledger

**Date:** 2026-07-12
**Scope:** Independent backend and frontend/Supabase repository tracks
**Release state:** P1.A and P1.B are locally complete, independently verified, and committed. Staging/production activation and live deployment evidence remain operator follow-ups.

## Repository identities

| Track | Repository/worktree | Branch | Evidence |
|---|---|---|---|
| P1.A backend | `D:/Enaiapp/langchain_railway` | `refactor/review-phase-fixes` | `5af3ff5` — `Publish P1 chat gateway contract` |
| P1.B frontend | `D:/Enaiapp/p0_frontend_commit_repo` (canonical Git checkout for the application linked to `D:/export_enai`) | `fix/p0-remediation` | `9f0bd6d` — canonical source; `05c0378` — deterministic auth; `7514282` — reproducible edge delivery |

The dirty/private legacy copies under `D:/export_enai` were not modified. Their retirement remains an operator action after the canonical deployment is proven.

## Package status

| Package | State | Verified evidence | Remaining action |
|---|---|---|---|
| P1.A gateway contract | Committed, locally verified | Versioned JSON/document contract, request-version mismatch handling, safe request correlation, full backend suite: 1,358 passed | Deploy `5af3ff5` or a descendant before the compatible chat edge function; record deployed backend SHA |
| P1.1 canonical source | Committed, locally verified | 132-difference SHA-256 reconciliation manifest; runtime/build/lint/CI/deploy paths reject the mirror; frontend tests/build passed | Point hosting at the canonical Git repository/commit, then archive private legacy copies |
| P1.2 deterministic auth | Committed, locally verified | Delayed `INITIAL_SESSION`, rapid sign-in/refresh/sign-out, stale profile/admin, timeout/offline, paused-user, protected-route, and cross-tab-style event tests pass | Run real two-tab and offline browser smoke tests after staging deployment |
| P1.3 deployable edge sources | Committed, locally verified | Standard nine-function layout; shared auth/CORS/log/error controls; exact dependency and Deno lock; cross-platform LF contract; Deno format/lint/check and 4 tests; 332 frontend/contract tests; lint/build; clean detached-checkout reproduction; source manifest `b571ad7161686abf1962d3839054adc8a159ad40ead5faf78246ff6c6c2b9321` | Push `7514282`, configure protected environments/secrets, deploy to staging by full SHA, verify all nine source headers and readiness, then promote the same SHA |

## P1.3 local completion evidence

The official Deno 2.1.4 Windows archive was verified against its published SHA-256 checksum before use. The generated Deno v4 lock, per-file SHA-256 manifest, and embedded aggregate identity are committed together. CI fails closed if the lock, manifest, or any deployable input is missing or stale.

Commit `75142826184c023dc13b5596286ac0afec0ab0e2` was checked out into a detached clean worktree and verified with:

```powershell
npm ci
npm run edge:lock
npm run edge:manifest:write
npm run edge:verify
npm test
npm run lint
npm run build
npm audit --omit=dev --audit-level=high
git diff --check
```

Results: 332 frontend/contract tests passed, 4 Deno tests passed, Deno format/lint/strict check passed, production build passed, and the production dependency audit reported zero vulnerabilities. Live Supabase/database checks were not required for the local commit and remain manual deployment attestations.

The detailed environment, secrets, deploy, proof, and rollback procedure is `docs/active/p1_edge_delivery_runbook.md` in the frontend repository.

## Activation order

1. Push the independent backend and frontend commits.
2. Deploy backend commit `5af3ff5` or a tested descendant and verify the `chat-gateway-v1` response header/correlation behavior.
3. Configure protected GitHub staging/production environments and Supabase runtime secrets.
4. Deploy frontend commit `7514282` (or a tested descendant with a regenerated manifest) by full 40-character SHA through the new workflow.
5. Prove all nine deployed `X-Enai-Edge-Source` identities, then run readiness and active/inactive/admin/authenticated staging smoke tests.
6. Deploy the frontend application from the same canonical repository and record its commit.
7. Promote the same edge SHA to production, retain evidence, and rehearse rollback to the previous approved SHA.

P3.B implementation may proceed from the reproducible source tree, but P3.B deployment must wait until P1.3's staging deployment identity and smoke tests are proven. P2.A may proceed independently because it does not depend on the edge delivery cutover.
