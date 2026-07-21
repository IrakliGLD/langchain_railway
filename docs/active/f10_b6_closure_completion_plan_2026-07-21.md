# F10 B6 Closure Completion Plan — 2026-07-21

Response to the independent re-audit
[`f10_b6_final_independent_closure_2026-07-21.md`](./f10_b6_final_independent_closure_2026-07-21.md)
(decision: **F10 not yet closed; do not merge the evidence branch yet**).

**Assessment: the audit is correct and fair. Every finding is accepted.** Two of
them are places the previous closure claim over-reached, and I own them:

- I wrote that the plan "accepts SHA-not-digest for Railway" — it does **not**.
  The governing B1.A control explicitly mandates the runtime image digest
  (`f10_blocker_remediation_plan_2026-07-18.md` lines 93, 94, 109, 117; line 118
  even anticipates the Railway-rebuild case — *"Record both digests explicitly"*).
  Substituting SHA + deployment ID needs a captured digest **or** a formally
  amended+approved control. Neither existed.
- The F3 matrix said item 6 `_pending_` while the prose said "carries forward,"
  and marked closure "done" with the operator list fully unchecked — an internal
  contradiction.

## 1. Findings assessment

| Finding | Sev | Verdict | Basis |
|---|---|---|---|
| **REL-02** runtime image digest absent | High | **Valid** | B1.A mandates the image digest (lines 93/94/109/117/118); the SHA+deployment-ID substitution had no waiver/amendment |
| **E2E-03** disposable-DB unclosed in ledger | High | **Valid** | matrix item 6 is literally `_pending_`; the carry-forward claim lacks #271's immutable run/artifact IDs + the source/hash-continuity proof binding it to `fc44fd4` |
| **A11Y-07** AT listening pass residual | Medium | **Valid** | structural pass is green, but the actual NVDA/VoiceOver *listening* attestation is a genuine operator residual, not evidence |
| **DOC-06** closure record self-contradicts | Medium | **Valid** | matrix/action-list/decision disagree with each other |
| **QA-09** brittle timing concurrency test | Low | **Resolved by auditor** | `tests/test_evidence_planner.py` now uses a `threading.Barrier(2)` overlap proof (uncommitted in the working tree); full suite 1710/1710 |

No finding is a false positive. The frozen live identities themselves
(backend `65cf93b`, frontend/Edge `fc44fd4`) are confirmed aligned — the gaps are
**evidence-completeness and the final merge/redeploy binding**, not defects.

## 2. Phased plan (follows the audit's "smallest safe completion sequence")

Ordered so the identity freezes/moves **once** (only at G5), and every gate is a
Pass or a complete waiver before closure.

### Phase G1 — Commit the QA-09 fix + reconcile the ledger (assistant) — [QA-09, DOC-06]

- Commit the barrier-based concurrency test (`tests/test_evidence_planner.py`) to
  the evidence branch; targeted suite green.
- Reconcile the F3 matrix + operator action list so they agree: item 6 →
  *carry-forward-with-proof* (G2) rather than `_pending_`; item 8 →
  *structural-done, SR-listen pending* (G4); remove any unconditional "closed"
  language. The action list is already relabeled "Historical — not a status
  ledger"; make the F3 runbook the single source of truth.

### Phase G2 — Close the disposable-DB evidence (assistant + operator) — [E2E-03]

Two acceptable routes; either satisfies the gate:
- **(a) Carry-forward with proof:** attach frontend CI **#271**'s immutable run
  URL/ID + its `test:db` step result + the byte-continuity proof
  (`git diff a4c24a2..fc44fd4 -- database/baseline database/tests` = empty; record
  both commit SHAs and both `git rev-parse` tree hashes for `database/`). This
  binds the #271 disposable-DB green to `fc44fd4`. **OR**
- **(b) Fresh stamp:** `TEST_DATABASE_URL` is now a GitHub secret, so `test:db`
  runs automatically on the next frontend CI at `fc44fd4` — record that run
  ID + green (throwaway `ufosbrdhjrkaagjaltno`, never prod).

### Phase G3 — Resolve the B1 runtime-digest control (assistant + operator) — [REL-02]

- **Capture the CI-attested image digest** from the Backend release evidence run
  `29761159168` artifact (`backend-release-manifest.json` image ID/revision) — this
  exists per B1.A line 93 and just needs recording in the ledger.
- **Attempt the Railway runtime image digest** from Railway build logs / API for
  deployment `ffc9ec32`.
- **If Railway exposes no immutable runtime digest:** draft a formal, dated **B1.A
  control amendment** for Irakli's approval — accept
  `{CI-attested image digest + Railway deployment ID + protected /versionz SHA +
  reproducible-build attestation}` as the immutable runtime identity for a
  source-building PaaS. Do **not** claim the unamended plan already accepts this.

### Phase G4 — AT listening attestation (operator) — [A11Y-07]

- Operator runs **NVDA** (Windows, free) or **VoiceOver** (~10 min) on the
  authenticated dashboard / chat / admin at `fc44fd4`. Record: browser + AT
  name/version, pages, keyboard flow, the chat-response / status announcements
  actually heard, and outcome. Assistant appends it to the F3 §6 a11y record.
  (If already performed elsewhere, just attach it — no rerun needed.)

### Phase G5 — Final merge → redeploy → re-verify at the final identity (operator + assistant) — [Final post-merge deployment]

Only after G1–G4 are green/attached:
- Merge `docs/f10-b6-f3-evidence` → `main`. Railway redeploys the **final merge
  SHA** (langsmith stays `0.10.7`; runtime closure unchanged).
- At that final identity, re-verify and record immutable IDs: installed
  `langsmith==0.10.7`; protected `/versionz git_sha` == final SHA; `/healthz`+
  `/readyz`; a fresh **Backend release evidence** run (+ image digest per G3);
  authenticated Post Deploy Smoke; and re-bind the rollback target
  (deployment + digest) to the final SHA.

### Phase G6 — Final B6 re-audit + closure (assistant) — [closure]

- Re-run the B6 closure matrix at the final identity; confirm no unwaived
  High/Critical; append the final immutable IDs; mark **F10 closed**.

## 3. Ownership & sequencing

| Phase | Assistant | Operator |
|---|---|---|
| G1 QA-09 + reconcile | commit + edit | — |
| G2 disposable-DB | write carry-forward proof / record CI run | (route b) let CI run |
| G3 runtime digest | extract CI digest, draft amendment | read Railway digest / approve amendment |
| G4 AT listening | append record | **run NVDA/VoiceOver** |
| G5 final merge/redeploy | verify + record IDs | merge, read Railway, `/versionz` |
| G6 re-audit | run matrix + close | final decision |

**Critical path:** G1 → (G2 ∥ G3 ∥ G4) → G5 → G6. G2/G3/G4 are independent and
can proceed in parallel. The only hard operator gates are G4 (screen reader), the
G3 Railway-digest read / amendment approval, and the G5 merge + `/versionz`.

## 4. Risk note

No new Critical/High code risk is introduced. G1 is a test-robustness commit; G2–
G4 are evidence capture; G5 is the deliberate single identity move the audit
requires, with full re-verification at the final SHA. The deployed runtime closure
is unchanged (`langsmith==0.10.7`, same 68 hashed pins).
