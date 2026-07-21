# F10 B6 — Historical Operator Dispatch Checklist — 2026-07-20

> This file preserves the original dispatch sequence; its checkboxes are not a
> current status ledger. Use
> [`f10_b6_f3_evidence_runbook_2026-07-20.md`](./f10_b6_f3_evidence_runbook_2026-07-20.md)
> and the latest independent closure audit for current pass/pending status.

Single "start here" checklist for everything left to close F10, in order.
All repo-side code/doc work (Phase F1) is done, audited, and pushed:
backend `refactor/review-phase-fixes` @ `c39cc65` (last code `e42833b`),
frontend `main` @ `d52a97e`. What remains is deploy + evidence + sign-off,
which needs your production access, a real screen reader, and a throwaway DB.

Legend: **[YOU]** operator action · **[ME]** assistant does it after you report back.

---

## Phase F2 — Freeze & aligned deployment

- [ ] **[YOU] O1 — Merge the backend F1 branch to `main` (merge commit).**
  Open `https://github.com/IrakliGLD/langchain_railway/compare/main...refactor/review-phase-fixes`,
  wait for CI green, **Create a merge commit** (as with B3 #88). Railway
  (`Enai`/`enerbot`) auto-deploys `main`.
- [ ] **[YOU] O2 — Capture the backend deployed identity.**
  Railway → Deployments (newest): record **image digest** `sha256:…`,
  **deployment ID**, and the **previous** deployment ID (= rollback target).
  Then `curl -H "X-App-Key: <gateway secret>" https://enai.galdava.com/versionz`
  and copy `git_sha`. → paste me: git_sha, image digest, deployment ID, rollback ID.
- [ ] **[YOU] O3 — Realign Edge to the browser's SHA.**
  GitHub → Actions → **Deploy Supabase edge functions** → Run workflow ·
  `git_ref = d52a97e3a3de34754444f5eeb02c2f3a9f1f5509` · `environment = production`.
  It self-verifies all nine functions report that version + matching source hash.
  → tell me "Edge green" (or the run link).
- [ ] **[ME] After O1–O3:** re-read all three surfaces and confirm they report
  **one** frozen identity (browser `d52a97e`, Edge `d52a97e`, backend merge SHA);
  fill the F2 evidence table and commit. **If any surface disagrees, F2 re-opens.**

## Phase F3 — Integrated evidence at the frozen identity

(Run only after F2 shows one identity. All are `workflow_dispatch`.)

- [ ] **[YOU] O4 — Frontend release evidence:** Actions → *Build frontend release
  evidence* · `git_ref = d52a97e3a3de34754444f5eeb02c2f3a9f1f5509` · `production`.
- [ ] **[YOU] O5 — Backend release evidence:** backend Actions → *Backend release
  evidence* · `git_ref = <O2 merge SHA>` · `production`.
- [ ] **[YOU] O6 — Authenticated Post Deploy Smoke:** Actions → *Post Deploy Smoke*
  · `production` (dispatch on `main`).
- [ ] **[YOU] O7 — Live Browser Proof + axe ×3:** Actions → *Live Browser Proof*
  (runs the authenticated proof **and** the login/public + dashboard/chat + admin
  axe scans).
- [ ] **[YOU] O8 — Disposable-DB regression:** re-run exactly as B4.B against a
  **throwaway** Postgres at the merge SHA — **never** production `qvmqmmcglqmhachqaezt`.
- [ ] **[YOU] O9 — Manual a11y checklist at `d52a97e`** (needs a real screen
  reader): authenticated visible-focus ring; NVDA name+role on tabs / chart /
  chat input / a toast; live-region announce on a chat reply or toast;
  responsive/touch at 375 & 768 px. Record browser+AT versions, pages, viewport.
  → *Optional [ME] assist:* if you log the in-app browser pane into the dashboard,
  I can run the programmatic authenticated a11y pass (DOM/axe-tree, focus order,
  accessible names, touch sizes) to cover everything except the live-SR listen,
  which only you can attest.
- [ ] **[ME] After each of O4–O9:** confirm the run/artifact targeted the frozen
  SHA, record run IDs + artifact names into the F3 table, and flag any drift.

## Phase F4 — Load & reliability closure

- [ ] **[ME] O10a — Finalize the §8 load waiver.** Once O2 gives the merge SHA,
  I fill the frozen backend+frontend SHAs into
  `f10_b6_load_waiver_2026-07-20.md`.
- [ ] **[YOU] O10b — Approve the waiver:** confirm approval + date (or, if you'd
  rather run the load envelope than waive it, say so and I'll switch it back).
- [ ] **[YOU] O11 — Rehearse rollback once:** in Railway, redeploy the O2 previous
  deployment, confirm `/healthz` + `/readyz` green, then roll forward; record the
  two deployment IDs + timings. → paste me the IDs.

## Phase F5 — Re-audit & closure

- [ ] **[ME] O12 — Re-run the B6 closure matrix** against the frozen immutable
  identities, confirm every gate is Pass or covered by a complete waiver, and
  assemble the final evidence ledger. F10 is marked closed only when the matrix
  has no unwaived High/Critical.

---

### Shortest critical path
`O1 → O2 → O3` (freeze) → `O4–O9` (evidence) → `O10–O11` (waiver + rollback) →
`O12` (my re-audit). O3 can run in parallel with O1/O2 (different repos). The
long poles are O7 (Playwright/axe run) and O9 (your screen-reader pass).
