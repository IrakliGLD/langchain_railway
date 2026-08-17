# Answer Consistency and Follow-Up Context — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILLS: `developer-phased-audit` (plan → implement →
> independent audit → fix, before the next phase) and `superpowers:test-driven-development`
> (red → verify red → minimal green → verify green). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Two questions that differ only in a company name should take the same path and come back
in the same register; and a short follow-up should reach the data its predecessor reached.

**Architecture:** Six defects from the 2026-08-17 10:48–10:55 session, in three groups. Phases 1–3
cut the chain by which a single analyzer coin-flip changes the whole answer, applying the rule this
codebase already relies on everywhere else — **the analyzer proposes, deterministic code decides**.
Phase 4 fixes a history-truncation direction bug. Phase 5 restores conversation context to the
analyzer, which is what actually broke the last question.

**Tech Stack:** Python 3.11 (container) / 3.14 (local), pandas, pytest.

## Global Constraints

- Targeted suite green **before each audit step**: `python -m pytest tests/ --ignore=tests/security -q`.
- `ruff check .` must pass.
- TDD mandatory for every phase.
- Do not change `strict_numeric` grounding thresholds, the summarizer budget, or the preview caps.
- **Lock the better answer**, by pinning what actually produced it. See the target profile below.
- Phases 3 and 5 change routing and retrieval inputs. Per the skill's core rule they need an
  explicit old-vs-new disagreement review before cutover, not just green tests.

---

## The Target Answer Profile — what "lock the better answer" means

The Telmico answer was better. This plan reproduces it deterministically, by pinning the three
things that **caused** it rather than the label that happened to accompany them.

| Property | Locked to | Where |
| --- | --- | --- |
| guidance focus | the richer regulation/tariff set (the 23,450-char one) | Phase 1 |
| `render_style` | `NARRATIVE` | Phase 2 |
| evidence plan | the single retail tool, no `get_tariffs` companion | Phase 3 |
| charts | the 3-chart comparison set | already deterministic |
| annual block, benchmark, widening | present | already deterministic |
| vector tier | `LIGHT` | already deterministic |

**Why not also pin `answer_kind=clarify`,** which is what the better run carried? Because after the
three rows above are pinned, it drives nothing — and pinning it would actively hurt:

1. **It is the wrong label.** `clarify` means "this question needs clarification". The question is a
   well-formed comparison. A label chosen for its side effects breaks the moment something reads it
   for its meaning.
2. **Three call sites treat `clarify` as an exception to rescue**, all added or touched this week:
   the vector tier (`agent/pipeline.py:720`), the chart selector
   (`visualization/chart_selector.py:78`), and the retail data-routing predicates. Pinning `clarify`
   makes the exception the normal case and inverts what those rescues mean.
3. **Verified: it changes nothing for this shape.** Every runtime branch on `answer_kind` outside
   the analyzer prompt keys on `FORECAST`, `EXPLANATION` or `LIST` — `agent/analyzer.py:2891/2919/
   2966/2970`, `agent/derived_chart_builder.py:1039/1084`, `agent/evidence_planner.py:160/462`,
   `agent/answer_mode_policy.py:66`. Neither `comparison` nor `clarify` reaches any of them. The
   remaining two consumers are already neutralised: the vector tier via `answers_from_data`, and the
   chart selector via the clarify-answered-from-data rescue — which is why **both** runs produced
   the identical 3-chart set.

So `answer_kind` is left free because it has been made inert for this shape, not because the answer
is left to vary. If Phase 1's audit shows the two question shapes still differ after the pins, that
assumption is wrong and this decision gets revisited.

---

## Evidence Base

Three requests, one session, 2026-08-17. All confirmed from the logs and by reading the code.

### The same question, twice, with only the company changed

| | EPS 10:48 (worse) | Telmico 10:52 (better) |
| --- | --- | --- |
| query_type / preferred_path | comparison / tool | ambiguous / knowledge |
| answer_kind / render_style | comparison / **deterministic** | clarify / **narrative** |
| 3rd candidate topic | balancing_price | exchange_transition |
| focus → guidance | general → 20,430 ch | **regulation → 23,450 ch** |
| evidence plan | 2 steps, `get_tariffs` **0 rows** + validator warning | 1 clean step |
| shipped answer | 2,265 ch | **3,689 ch** |

Identical in both: the 3-chart set, the annual make-or-buy block, the benchmark, the widening.
Those paths ignore the analyzer, which is exactly why they did not vary.

### The variance is a cascade, not a direct effect

```
analyzer non-determinism
  -> candidate_topics differ
  -> preferred_topics differ            (fed into vector retrieval)
  -> retrieval ranking differs          (same two top_sources, opposite order)
  -> retrieved document_type set differs
  -> focus general vs regulation        (core/llm.py:6403-6410)
  -> guidance 20,430 vs 23,450 chars
  -> visibly different answer
```

`get_query_focus()` (`core/query_classifier.py:100`) is **deterministic on the query text** and
returns `"general"` for both — its tariff branch needs `ტარიფი` and the question says `ფასი`. So
focus is decided entirely by the retrieval-driven fallback.

**Root confirmed:** the Telmico question at 10:23 and at 10:52 carried
`prompt_sha256=ed17c5c40eaa`, the same query hash, and `cached_prompt_tokens=11010` — byte-identical
prompt, cache hit — yet returned `comparison/tool/0.85` versus `ambiguous/knowledge/0.78`.

### `render_style=deterministic` was wrong, and the system already knew

The EPS run logged
`Plan validation: render_style=DETERMINISTIC but plan has 1 narrative-augmentation step(s)
(['tariff_context'])` and continued anyway. A make-or-buy answer carries irreversibility,
seasonality, load-shape and regulatory-cycle caveats; a deterministic renderer cannot express them.

### The follow-up that could not reach the data

Request 3, 10:54:54: `candidate_topics=[]`, `candidate_tools=[]`, `entity_scope=None`,
`confidence=0.15`, `canonical_query_en` 68 chars → `KNOWLEDGE_PRIMARY`,
`tool_blocked_by_policy=true`, **`data_preview_chars=0`**, `statistics_chars=283`. Answered from
24 KB of knowledge prose with no data at all.

Cause, at `core/llm.py:3385`:

```python
needs_history = bool(history_str) and (prompt_profile == "clarify" or has_anaphoric)
```

The analyzer receives conversation history **only** when `_ANAPHORIC_HISTORY_RE` matches or the
profile is already clarify. That regex carries four Georgian phrases (`იგივე`, `და ასევე`,
`რაც შეეხება`, `ესეც`). An **elliptical** follow-up — "და 2023 წელს?" — contains no anaphor at all,
so the analyzer was handed a bare fragment with no context.

Note the asymmetry: `conversation_history_chars=1707` — the **summarizer** got the history. The
stage that needed it to route did not.

### History truncation runs in opposite directions

- `utils/session_memory.py:537` (seeding) — `turns[:SESSION_HISTORY_MAX_TURNS]` keeps the **oldest** N
- `utils/session_memory.py:570` (append) — `del history[:-SESSION_HISTORY_MAX_TURNS]` keeps the **newest** N

With the cap at 2 and three caller turns supplied, seeding persists turns 1–2 and discards turn 3 —
the most recent, which is the one a follow-up refers to. The logged `turns=3` is
`len(bound_history)`, the caller list used for that turn, not the capped copy that persists.

---

## Phase 1 — Make the guidance focus deterministic

**Goal:** the same question shape selects the same guidance set, whatever retrieval returned.

- [ ] Failing test: for a retail make-or-buy frame, the resolved focus is the same whether the
      retrieved chunks carry a regulation-ish `document_type` or not.
- [ ] Failing test: the doc-type fallback is **order-independent** — the same set of document types
      yields the same focus regardless of iteration order. `core/llm.py:6406` currently does
      `for dt in doc_types: ... break` over a **set**, so the winner depends on set ordering rather
      than on any stated priority. Replace with an explicit priority sequence.
- [ ] Failing test: a question whose text *does* determine a focus (contains `ტარიფი`) still wins
      over the fallback — `get_query_focus` stays authoritative when it is not "general".
- [ ] Implement: for a recognised make-or-buy retail frame, select the focus deterministically
      rather than inheriting it from retrieval ranking. Reuse the predicate that already identifies
      the frame elsewhere rather than adding a fourth spelling of it.
- [ ] Audit: measure the guidance size for both the EPS and Telmico question shapes and confirm they
      now match. Record both numbers; a residual difference means something else still varies.

## Phase 2 — Pin the render style for a make-or-buy frame

**Goal:** the answer is written as an assessment, every time.

- [ ] Failing test: a make-or-buy retail frame resolves to `RenderStyle.NARRATIVE` even when the
      analyzer emitted `DETERMINISTIC`.
- [ ] Failing test: a genuinely deterministic shape (a scalar lookup) is untouched.
- [ ] Implement using the existing idiom — `agent/evidence_finalizer.py:261-264` already sets
      `ctx.question_analysis.render_style = RenderStyle.NARRATIVE` on an uncorrectable evidence gap,
      so this is an established override point, not a new mechanism.
- [ ] Consider promoting the existing plan-validation warning into this decision, so the system acts
      on a mismatch it already detects instead of only logging it.
- [ ] Audit: check every downstream reader of `render_style` (`agent/pipeline.py:678` tier
      resolution, `agent/evidence_planner.py:463`, `agent/contract_continuity.py:52`,
      `agent/fixture_candidates.py:34`) for a behaviour change the tests do not cover.

## Phase 3 — Drop the tariff companion step on a retail primary (routing change)

**Goal:** stop planning a step that has returned 0 rows in every observed run.

- [ ] Failing test: with `get_end_user_prices` as the primary tool, an analyzer-emitted
      `tariff_context` role plans **no** `get_tariffs` step.
- [ ] Failing test: with `get_prices` (wholesale) as the primary, `tariff_context` still maps to
      `get_tariffs` — the role is meaningful there and must keep working.
- [ ] Implement in `_role_to_default_tool` (`agent/evidence_planner.py:632`), which already varies
      the mapping by primary tool. `tariff_context` means "regulated tariff series as context"; on a
      retail frame the primary data *is* the tariff series, and `get_tariffs` is the generation-side
      table.
- [ ] **Disagreement review before cutover:** run old and new planners over the retail and wholesale
      questions in `evaluation/` plus the traces on record, diff the planned tool sets, and inspect
      every difference by hand.
- [ ] Audit: confirm the plan-validation warning about narrative-augmentation steps stops firing on
      this shape, and that nothing else depended on the step existing.

## Phase 4 — Seed session history from the tail

**Goal:** the most recent turn survives seeding.

- [ ] Failing test: seeding with more caller turns than `SESSION_HISTORY_MAX_TURNS` keeps the
      **newest** N, not the oldest.
- [ ] Failing test: seeding with fewer turns than the cap keeps all of them, unchanged.
- [ ] Failing test: seeding and appending agree — the same sequence of turns through either path
      leaves the same stored history.
- [ ] Implement: `utils/session_memory.py:537` becomes `turns[-SESSION_HISTORY_MAX_TURNS:]`.
- [ ] Audit: check whether the `turns=%d` logged at `main.py:1535` should report the stored count
      rather than `len(bound_history)`. It read 3 under a cap of 2, which is true but reads as a cap
      violation and cost time to diagnose.

## Phase 5 — Give the analyzer the conversation (routing change)

**Goal:** a short follow-up resolves against its predecessor instead of arriving as a bare fragment.

- [ ] Failing test: an **elliptical** follow-up with no anaphor ("და 2023 წელს?") still results in
      conversation history being included in the analyzer prompt.
- [ ] Failing test: with no session history, the prompt is unchanged — no empty history block.
- [ ] Implement: drop the anaphora gate at `core/llm.py:3385` so history is included whenever the
      session has any. Cost is ~1.7 KB against a live analyzer prompt of ~46.4 KB with an effective
      budget of 63.9 KB, so there is ample headroom; the gate's failure mode is silent and total.
- [ ] Failing test: **carry-forward.** When the analyzer returns a structurally empty analysis
      (`candidate_topics == []` **and** `candidate_tools == []` **and** `entity_scope is None`) and
      the session has a previous turn, the prior turn's resolved scope is reused instead of routing
      to `KNOWLEDGE_PRIMARY` with no data. Key on structural emptiness, not on a confidence
      threshold — thresholds are brittle and 0.15 is not a documented boundary.
- [ ] Investigate first, then decide the home: `agent/contract_continuity.py` and
      `utils/session_memory.set_last_contract` already snapshot the previous turn's contract, and
      `ENABLE_CONTRACT_CONTINUITY` gates it. Verify whether that mechanism is enabled in production
      and why it did not cover this case, before adding a parallel one.
- [ ] **Disagreement review before cutover:** the anaphora gate currently suppresses history on most
      turns, so removing it changes the analyzer's input on nearly every multi-turn request. Compare
      classifications with and without history over the session's questions and inspect the
      differences; more context is not automatically better.
- [ ] Audit: confirm the follow-up now reaches `get_end_user_prices` and that
      `data_preview_chars > 0`, which is the observable that was zero.

---

## Deliberately Not Doing

- **Pinning `answer_kind` to either value.** Not because the answer should vary — Phases 1–3 lock
  it — but because after those pins the field drives nothing for this shape, `comparison` is the
  semantically right label, and `clarify` is read as an exception by three rescue call sites. See
  "The Target Answer Profile" above for the reasoning and the verification.
- **Sampling determinism as the primary fix.** Reasoning models frequently ignore `temperature`, the
  `sampling_temperature` plumbing (`core/llm.py:540`) is unset for this stage, and it would not
  touch the retrieval-ranking or focus-fallback links in the cascade. Reconsider only if variance
  survives Phases 1–3.
- **Broadening the anaphora regex.** Phase 5 removes the gate instead. Extending a phrase list is
  the same unbounded-wording approach that has failed four times on the retail path, by
  `retail_routing.py`'s own account.

## Open Question for the Domain Owner

Phase 2 pins `NARRATIVE` for make-or-buy frames on the evidence of one better answer plus the plan
validator's own objection. If any retail comparison should render deterministically — a bare
"what is the spread this month" — say so before cutover and the pin gets a narrower trigger.
