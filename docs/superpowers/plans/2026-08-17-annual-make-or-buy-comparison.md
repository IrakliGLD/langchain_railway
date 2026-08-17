# Make-or-Buy Comparison: Annual Evidence, Preview Fidelity, Knowledge Reach

> **For agentic workers:** REQUIRED SUB-SKILLS: `developer-phased-audit` (phase discipline: plan →
> implement → independent audit → fix, before the next phase) and `superpowers:test-driven-development`
> (red → verify red → minimal green → verify green). Steps use checkbox (`- [ ]`) syntax.

**Goal:** When a retail question sets the regulated supply tariff against the wholesale benchmark,
the answer states which side was cheaper **in each year and by how much**, in the units the domain
owner uses, with the regulatory-cycle caveat that makes a single year's sign readable — instead of
a multi-year average that averages the answer away.

**Architecture:** Six phases. Phases 1–3 deliver the answer itself: a deterministic per-year block
in the grounding corpus, plus the two content rules needed to read it. Phases 4–5 restore the data
and knowledge that the prompt was silently losing. Phase 6 is the routing change and comes last
because it alters which rows get fetched.

**Tech Stack:** Python 3.11 (container) / 3.14 (local), pandas, pytest.

## Global Constraints

- Targeted suite green **before each audit step**: `python -m pytest tests/ -q` from the backend
  dir. "Green on the modules I changed" is not sufficient (`developer-phased-audit`
  `references/workflow.md` §6).
- `ruff check .` must pass.
- TDD is mandatory for every phase that touches production code (1, 2, 4, 5, 6). Phase 3 is
  content; its tests assert load-bearing facts, never the presence of a paragraph.
- Do not change `strict_numeric` grounding thresholds. The gate is behaving correctly; the corpus
  it reads is the defect.
- Never pool across series. An average across suppliers or categories is a price nobody pays.
- Phase 6 changes routing. Per the skill's core rule, it requires an explicit old-vs-new
  disagreement review before cutover, not just green tests.
- Record per-phase output in the shape the skill requires: goal, implementation summary, **audit
  findings first**, fixes applied, remaining risks, next-phase recommendation.

---

## Evidence Base

From the 2026-08-16 production trace `span-20025d51e30049e092b5ee59d7157045` (Georgian query,
Telmico, make-or-buy comparison). All confirmed by reading code, not inferred from the log:

| Observation | Value | Source |
| --- | --- | --- |
| Tool fired with the benchmark | 7 numeric cols = 5 retail + benchmark + spread | `end_user_price_tools.py:455` |
| Rows returned | 528 = 66 months × 8 series | trace `stage_0_6_tool_execute` |
| After `rows_to_preview` | 528 → 124 rows, **no omission marker emitted** | `analysis/stats.py:96-101` |
| After summarizer compaction | 124 → 60 rows (head 62% / tail) | `core/llm.py:6040` |
| Net visible history | ~5 newest months + ~3 oldest; **2022–2025 absent entirely** | `ORDER BY date DESC` + head/tail |
| Comparison figures available | per-series **whole-period** mean/min/max only | `agent/analyzer.py:3358` |
| Domain knowledge in prompt | **0 chars** | trace `summarizer_prompt_census` |
| Numeric claims shipped | 3, in 2,470 chars | trace `provenance_gate` |

The model behaved correctly given its inputs. A single 2021–2026 mean spread per category **by
construction** hides the year-to-year sign flips the question is about.

Four mechanical facts that constrain the design:

1. `is_intensive_metric("supply_vs_wholesale_spread_gel_kwh")` returns **False** — the column
   carries the extensive token `supply` and no intensive token, so `_describe` emits a `sum=` for
   it: a per-kWh spread summed across 66 months, live in the grounding corpus today.
   (`supply_tariff_gel_kwh` is safe — `tariff` is intensive and intensive wins.)
2. `_add_rounded_source_variants` (`agent/summary_grounding.py:297`) emits rounded forms of a
   token but never a ×1000 form. A corpus containing `0.1450` does **not** make `145` quotable.
3. `rows_to_preview` concatenates head and tail with **no marker**, so a truncated series reads as
   contiguous: `2026-06` on one line, `2021-09` on the next.
4. Three label columns dominate preview row width, repeated on all 528 rows:
   `supply_company` = `Telmico (Tbilisi Electricity Supply Company)` (43 ch), `series_label` =
   `Telmico — Commercial - other (3.3-6-10)` (38 ch), `category_label` up to 37 ch — about
   **55–60% of each ~197-char row**.

---

## Block Design

### Trigger

Emit when the frame carries **both** `supply_tariff_gel_kwh` and `wholesale_benchmark_gel_kwh`,
and a time column resolves. Absent either, emit nothing — this is the make-or-buy shape, not a
general per-year facility.

### Sign convention (load-bearing — test both directions)

`spread = supply_tariff − wholesale_benchmark`, matching the tool's own
`supply_vs_wholesale_spread_gel_kwh`.

- `spread < 0` → tariff below benchmark → **regulated cheaper**
- `spread > 0` → tariff above benchmark → **wholesale cheaper**

Against the domain owner's figures: 2022 regulated 145 vs wholesale 147 → −2 → regulated cheaper.
2024 regulated 170 vs wholesale 168 → +2 → wholesale cheaper.

### Coverage rule (domain owner, 2026-08-17)

Partial years are included and reported as partial. The report names the **months**, not just the
count, for two reasons:

1. The wholesale side is seasonal, the regulated tariff is flat (`retail-tariff-rules.md` states
   this asymmetry). A Jul–Dec partial and a Jan–Jun partial are biased in opposite directions.
   `currency_influence.md` §7 independently prescribes comparing "yearly averages, or same
   seasonal periods across years".
2. The benchmark arrives via `LEFT JOIN` on `price_with_usd`, so a month can carry a tariff with a
   null benchmark. Coverage is counted on **paired** months — both sides present — not on tariff
   months.

### Emitted shape

```
--- ANNUAL MAKE-OR-BUY COMPARISON ---
Regulated supply tariff vs like-for-like wholesale benchmark (balancing price + guaranteed
capacity fee + ESCO service fee), averaged per calendar year, per series. Figures are per series;
do not average across them. A PARTIAL year is computed only from the months listed: the wholesale
side is seasonal and the tariff is not, so a partial year is not directly comparable to a full one.

[supplier=telmico, category=3.3-6-10|com|other]
  2021 PARTIAL 6/12 (Jul-Dec)  supply=0.1450 (145.0 GEL/MWh)  benchmark=0.1470 (147.0)  spread=-0.0020 (-2.0)  regulated cheaper
  2022 FULL    12/12           supply=0.1450 (145.0 GEL/MWh)  benchmark=0.1470 (147.0)  spread=-0.0020 (-2.0)  regulated cheaper
  2024 FULL    12/12           supply=0.1700 (170.0 GEL/MWh)  benchmark=0.1680 (168.0)  spread=+0.0020 (+2.0)  wholesale cheaper
  Full years: 4. Regulated cheaper in 3, wholesale cheaper in 1.
```

- Both unit renderings on every value line — GEL/kWh at 4 dp (stored unit) and GEL/MWh at 1 dp
  (the unit the comparison is discussed in). Required by fact 2 above.
- The per-series footer counts **full years only**, so a 6-month stub cannot swing the headline.

### Size budget

8 series × 6 years ≈ 48 lines ≈ 6 KB on an 11.7 KB `stats_hint`. `_MAX_ENUMERATED_SERIES = 20` is
too permissive (20 × 6 = 120 lines ≈ 15 KB). Cap on **emitted rows**: enumerate while
`series_count × year_count <= 60`; above that emit the per-year cross-series tally only, with the
instruction to scope the question.

---

## Phase 1 — Metric classification

**Goal:** the spread column is averaged, never summed. Prerequisite: Phase 2 averages it, and the
existing whole-period block sums it today.

- [ ] Failing test: `is_intensive_metric("supply_vs_wholesale_spread_gel_kwh")` is True.
- [ ] Failing test: the existing column-aggregates block emits no `sum=` for that column.
- [ ] Fix in `analysis/stats.py:28` — add `spread` and `benchmark` to `_INTENSIVE_TOKENS`. Prefer
      this to a column-name special case: intensive tokens already take precedence, so this
      restores intended precedence rather than adding an exception.
- [ ] Regression guard: `supply_quantity`, `generation`, `demand`, `consumption`, `export`,
      `import` still classify as extensive.
- [ ] Audit: confirm no other column in any tool's output flips classification unintentionally.
      Grep the tool modules for numeric column names containing `spread`/`benchmark`.

## Phase 2 — Annual comparison block

**Goal:** per-year comparison figures in the grounding corpus, surviving compaction.

- [ ] Failing test: two-year single-series frame → one line per year, correct verdict, both unit
      renderings present.
- [ ] Failing test: **sign convention, both directions.** Tariff below benchmark → `regulated
      cheaper`; above → `wholesale cheaper`. An inverted verdict is the highest-consequence
      failure this block can have.
- [ ] Failing test: a 6-month year is marked `PARTIAL`, names its month span, and is excluded from
      the full-year tally while still appearing as a line.
- [ ] Failing test: a month with a tariff but null benchmark is excluded from that year's paired
      coverage count and from both means.
- [ ] Failing test: frame lacking `wholesale_benchmark_gel_kwh` emits no block.
- [ ] Failing test: above the 60-row budget, degrades to the cross-series tally with no per-series
      detail.
- [ ] Implement `_append_annual_comparison(ctx)` in `agent/analyzer.py`, reusing
      `_series_key_columns` (`:3242`) and `_time_column` (`:3143`). Call from `:2722`, immediately
      after `_append_column_aggregates(ctx)`.
- [ ] Means use `.mean()` over non-null paired months. Never `.sum()`, regardless of what
      `is_intensive_metric` returns — Phase 1 makes the classifier agree, but this block must not
      depend on it.
- [ ] Add `"ANNUAL MAKE-OR-BUY COMPARISON": 88` to `_SUMMARIZER_STATS_SECTION_PRIORITY`
      (`core/llm.py:5975`). Without an entry it defaults to 10 — below `COLUMN AGGREGATES` at 25 —
      and is shed first, reproducing the defect this plan exists to fix. 88 sits above
      `CORRELATION MATRIX` (80), below `DERIVED ANALYSIS EVIDENCE` (85).
- [ ] Failing test: the header matches `_SUMMARIZER_STATS_SECTION_RE` and resolves to 88, not the
      default. The regex requires the exact `--- NAME ---` form on its own line.
- [ ] Failing test: with `stats_hint` over `max_chars`, the annual block survives and
      `COLUMN AGGREGATES` is shed first.
- [ ] Audit: verify emitted numbers are byte-identical to what the grounding tokenizer will accept
      (`_extract_number_tokens` + `_add_rounded_source_variants`), for both `0.1450` and `145.0`.

## Phase 3 — Knowledge and guidance

**Goal:** the two content rules that make the block readable, at minimum surface. Content phase;
tests assert facts, never paragraphs.

**Fact 3 — regulatory-period true-up. Minimal treatment, two edits, no new sections.**

- [ ] `knowledge/network_supply_tariffs.md` § *Relationship to the wholesale market* — extend the
      existing bullet list with **one** bullet giving the mechanism: the supply tariff is fixed for
      the regulatory period against an *expected* wholesale price; a shortfall or surplus is
      compensated in the **next** period, so the regulated price can sit above wholesale while it
      recovers an earlier period's loss, and below it after the reverse. Cross-reference
      `tariffs.md`, which already carries the analogous generation-side lag ("reflected only in the
      next regulatory period"). Note this bites hardest on the `public` supply activity —
      `com|other` at each of the three voltages, 3 of the 8 categories.
- [ ] `skills/energy-analyst/references/retail-tariff-rules.md` § *Comparing against the wholesale
      price* — add the **reading rule** (this is the load-bearing half, and the guidance path is
      the one that actually reached the model in the trace): one year's sign is not a verdict on
      that year's market, because it may be recovering an earlier period. Do not present a single
      year's reversal as evidence that one side is structurally cheaper.

**Fact 2 — household structural gap.**

- [ ] `knowledge/network_supply_tariffs.md`, under the comparison section. Approved wording:
      "Household regulated prices sit well below the wholesale benchmark, and under current
      arrangements they stay there: the universal supplier serving households has access to
      low-cost sources that other suppliers do not." Phrase as a standing scoped statement, not a
      prediction — `tests/test_knowledge_freshness.py` guards that convention, and "under current
      arrangements" matches the file's existing "only under the current transitional model".
      Attribute as domain-owner-supplied, as the ESCO constant already is.
- [ ] `retail-tariff-rules.md` — the answering consequence: for household categories the comparison
      has a known answer; state it in one line and spend the analysis on the commercial categories.
      This is where the household narrowing lives, **not** in tool selection.

**Fact 1b — FX. One line only.**

- [ ] `network_supply_tariffs.md` comparison section: moving to wholesale takes on exchange-rate
      risk on top of electricity-market risk, which the regulated tariff does not carry within the
      regulatory period. Pointer to `currency_influence.md`. **Do not** reproduce the channel list
      or the mechanism, and **do not** edit `currency_influence.md` — §2.1 and §6 already carry it
      in full (domain owner, 2026-08-17).

**Guidance reconciliation.**

- [ ] `retail-tariff-rules.md` currently says "compare over a **sustained period**, not month by
      month" and "never write a month-by-month strategy". Read literally that argues against the
      new block and the model may suppress it. Narrow the rule's scope: a switching **strategy**
      across periods is what does not exist and stays forbidden; a per-year **record of outcomes**
      is the volatility evidence an irreversible decision needs. The existing bullet "treat the
      month-level detail as volatility evidence" already licenses this — make it explicit at the
      annual grain.
- [ ] Add: when the annual block is present, state the per-year figures and verdict, then the
      bottom line over the full horizon. Quote the block; do not recompute.
- [ ] Add: a `PARTIAL` year is reported as partial with its month span, and not compared
      like-for-like with a full year.
- [ ] Tests assert the load-bearing facts: partial-year reporting required, per-year verdicts
      permitted, switching strategies still forbidden, single-year sign not a structural verdict.
- [ ] Audit: re-read both files end-to-end for internal contradiction. These two documents overlap
      heavily on the comparison and must not disagree.

## Phase 4 — Preview fidelity

**Goal:** stop losing history silently. **All three changes activate only when the preview would
otherwise be truncated**, so small frames are untouched and the blast radius is bounded.

- [ ] Failing test: a truncated preview contains an explicit omission marker. Fix
      `analysis/stats.py:96-101` — `pd.concat([head, tail])` currently emits no marker, so the
      model cannot tell rows were dropped. Reuse the wording already used downstream
      (`...[middle preview rows omitted]...`) so the two stages read consistently.
- [ ] Failing test: when truncation is needed and a low-cardinality long-text column is
      functionally determined by a retained column, it is dropped from the rows and emitted once as
      a legend above the CSV. Keep the rule general — a non-numeric column whose values are
      constant within each group of a retained column, above a length threshold — not a hardcoded
      list of retail column names.
- [ ] Failing test: the legend preserves the ability to name companies in full. `retail-tariff-rules.md`
      says "The `supply_company` column already holds the full name — quote it rather than the
      `supplier` code"; update that sentence to point at the legend, in the same change.
- [ ] Failing test: when truncation is needed and the frame spans more than one year, rows are
      sampled by **date** — evenly spaced across the span, always including the first and last date
      — and **all series are retained at every sampled date**. Sampling by row position would break
      cross-series comparability at a given date; sampling by date does not.
- [ ] Failing test: a frame under budget is byte-identical to today's output.
- [ ] Audit: measure actual row yield on a 528-row, 8-series, 66-month frame before and after.
      Expected roughly 60 → ~140 rows spread across 2021–2026. Record the measured number; if the
      gain is materially smaller, re-plan rather than proceed.

## Phase 5 — Vector tier

**Goal:** `network_supply_tariffs.md` reaches the prompt when the pipeline answers a
clarify-shaped question from data. Trace showed `domain_knowledge_chars=0`.

- [ ] Failing test: `answer_kind=CLARIFY` with a data-backed retail question resolves to `LIGHT`,
      not `SKIP`.
- [ ] Failing test: `answer_kind=CLARIFY` on a genuinely unanswerable question still resolves to
      `SKIP`. The rescue must not disable clarify handling generally.
- [ ] Implement: add an `answers_from_data: bool = False` parameter to
      `_resolve_vector_retrieval_tier` (`agent/pipeline.py:676`); when True, `CLARIFY` falls
      through to the data-shape branches instead of short-circuiting. `_resolve_vector_tier`
      (`:2576`) computes it as `is_retail_data_question(ctx) or is_data_backed_ambiguous_question(ctx)`.
      Both are already imported at `:66-68` and are pure functions of ctx — no new coupling.
- [ ] Rationale to record in the docstring: the CLARIFY→SKIP premise ("no data to ground, no
      knowledge to cite") is stale for exactly the questions `retail_routing` overrides. Tying the
      tier to the same predicate that flips the route stops the two disagreeing again — the same
      class of bug `retail_routing.py` was created to fix.
- [ ] **Rejected alternative, recorded:** reordering `_apply_response_mode` before
      `_resolve_vector_tier` is cleaner in principle but `_apply_response_mode` *writes*
      `ctx.is_conceptual` while the tier resolver *reads* it, so reordering silently changes the
      `is_conceptual` rescue for every query. Worth doing as its own change, not riding along.
- [ ] Audit: confirm the chart skip (`answer_kind=clarify`) is a separate code path and note
      whether it deserves the same treatment. Do not fix it here.

## Phase 6 — Scope resolution (routing change — disagreement review required)

**Goal:** stop discarding successfully-extracted scope. `_compose_category`
(`end_user_price_tools.py:191`) extracts voltage and class, then returns `None` if either is
missing — which widened a 6–10 question from 2 categories to all 8.

- [ ] Failing test: text naming a voltage but no class resolves to the categories **at that
      voltage** (2 at `3.3-6-10`), not all 8 and not one guessed category.
- [ ] Failing test: text naming neither still widens to all 8 — unchanged.
- [ ] Implement additively: a `resolve_voltage(text)` helper plus a `voltage` parameter on
      `_resolve_selection` / `get_end_user_prices`; the planner sets it when category is unresolved
      but voltage resolved. Additive so existing return types are unchanged.
- [ ] Check `agent/tools/capabilities.py` and `tests/test_capability_registry_completeness.py` —
      a new tool parameter may need registry updates. Contract work before integration.
- [ ] Failing test: `6-10 კვტ` (kW — connection power) does **not** resolve a voltage. A number
      adjacent to a power unit is not a voltage. Today `"6-10"` is a kV alias and only the missing
      class word prevented a confidently wrong voltage pick.
- [ ] Implement unit-aware matching: match a bare numeric voltage only when a voltage unit is
      adjacent (`kv`, `კვ`, volt); refuse when the adjacent unit is a power unit (`kw`, `კვტ`,
      `kva`).
- [ ] Failing test: en-dash variants resolve. The analyzer emitted `6–10`; only the Georgian
      original's plain hyphen matched. Add en-dash forms for every numeric alias.
- [ ] **Disagreement review before cutover** (skill core rule for routing changes): run the old and
      new resolvers over the retail questions in `evaluation/` plus the traces on record, diff the
      resolved `(supplier, category, voltage)` triples, and review every disagreement by hand. Green
      tests are not sufficient for a routing change.
- [ ] Audit: confirm no question that previously resolved to a single category now widens, and no
      question that previously widened now resolves to a *wrong* single category. The second is the
      dangerous direction.

---

## Cross-cutting finding (Phase 3 audit, 2026-08-17): the summarizer prompt budget now binds

Measured, not estimated:

| Section | Trace 2026-08-16 | After Phases 2-3 | Delta |
| --- | --- | --- | --- |
| statistics | 11,686 | 19,013 | **+7,327** (annual block, 8-series frame) |
| guidance | 14,149 | 16,945 | **+2,796** (retail rules 9,052 -> 11,848) |
| **total prompt** | **41,175** | **~51,298** | **+10,123** |

`SUMMARIZER_PROMPT_BUDGET_MAX_CHARS` defaults to `PROMPT_BUDGET_MAX_CHARS` = **45,000**
(`config.py:631`) and production does not override it. The trace ran at 41,175 with **3,825 to
spare**; after Phases 2-3 the same query lands roughly **6,300 over**.

`_TRUNCATION_PRIORITY_DATA` sheds in this order:
`CONVERSATION_HISTORY` -> `DOMAIN_KNOWLEDGE` -> `EXTERNAL_SOURCE_PASSAGES` -> `DATA_PREVIEW` ->
`STATISTICS`.

Consequences for the phases still to come:

- **The annual block is safe.** `STATISTICS` is shed last, and inside it the block holds priority
  88. The Phase 2 goal survives the squeeze.
- **Phase 5 is at risk of being neutralised.** `DOMAIN_KNOWLEDGE` is shed **second**. Restoring
  `network_supply_tariffs.md` to the prompt only to have the budget cut it again would leave
  `domain_knowledge_chars=0` for a different reason. Phase 5 must verify the knowledge actually
  survives budgeting, not merely that the tier resolves to LIGHT.
- **Phase 4 is partly clawed back.** `DATA_PREVIEW` is shed fourth, so roughly 6,300 chars would
  come out of the ~11,944-char preview — about half the rows. Phase 4's row-density work
  (legend-once) directly offsets this and becomes more valuable, not less.

**Required deployment action, outside the code change:** raise
`SUMMARIZER_PROMPT_BUDGET_MAX_CHARS` (71,000 matches what is already set for the analyzer knob).
Until that is set, the phases below are competing for a budget that is already exceeded. This is
the summarizer knob previously identified as outstanding.

### Confirmed by measurement in the Phase 5 audit

`_section_aware_truncate` run over post-Phase-4 section sizes (prompt 52,850 chars), with the
Phase 5 tier fix in place so `DOMAIN_KNOWLEDGE` is actually retrieved at LIGHT (~4,000 chars):

| budget | domain knowledge | data preview | statistics |
| --- | --- | --- | --- |
| **45,000 (current default)** | **0 — fully shed** | 8,830 (of 11,833) | 19,013 intact |
| 55,000 | 4,000 intact | 11,833 intact | 19,013 intact |
| 71,000 | 4,000 intact | 11,833 intact | 19,013 intact |

So at the shipped default **Phase 5 has no effect**: the tier resolves to LIGHT, the passages are
retrieved, and the budget then discards all of them — `domain_knowledge_chars=0` again, for a
different reason than the trace. Phase 4's preview also loses ~3,000 of its ~11,833 chars, about a
quarter of the gain.

The annual block is unaffected at every budget, so Phase 2 stands on its own.

Anything at or above ~53,000 is sufficient; 71,000 is recommended only for consistency with the
analyzer knob and headroom. This is a cost/latency decision (larger prompts, more tokens), so it
is left as an operator action rather than a changed default. It is **observable** without new
logging: the existing `summarizer_prompt_census phase=post_budget` line reports
`domain_knowledge_chars`, and a pre/post mismatch is the signal.

## Phase 6 disagreement review (routing cutover gate, 2026-08-17)

Old and new resolvers run over 18 retail phrasings — the trace wording, the clarification example
the guidance itself offers, Georgian forms, en-dash forms, class-only and voltage-only forms, and
false-positive checks. **3 disagreements of 18**, each inspected:

| Query | Old | New | Verdict |
| --- | --- | --- | --- |
| `telmico customer at 6-10 kv` | 8 categories | 2 at `3.3-6-10` | **Intended** — the defect this phase exists to fix |
| `telmico 6-10 kw commercial customer` | **1 category, the wrong voltage** | 4 commercial | **Strict improvement** — a confidently-wrong single category becomes a correct superset containing the true one |
| `high voltage` | 16 (all) | 2 at `35-110` | **Intended** — a stated voltage should scope |

The dangerous direction — a question that previously resolved to one category now resolving to a
DIFFERENT single category — does not occur. Every change is either a widen-from-wrong or a
narrow-from-unscoped.

Checked and deliberately unchanged: `თელმიკო, 6-10 კვტ` resolves no supplier in either version
(Georgian company names are not in `SUPPLIER_ALIASES`). This does not bite in production because
the haystack includes the analyzer's English `entity_scope` — "Telmico customer in stated 6–10
kW/kV category" — which is where the supplier actually resolved on the trace. Georgian supplier
aliases remain out of scope.

`high-voltage` (hyphenated, as in "high-voltage transmission network") does **not** match the
`high voltage` alias, because `_alias_matches` bounds on non-alphanumerics — so the transmission
phrasing is unaffected.

## Out of Scope

- **Chart suppression on `answer_kind=clarify`.** Noted in Phase 5's audit; not fixed here.
- **Reordering `_apply_response_mode` / `_resolve_vector_tier`.** Recorded as a rejected
  alternative in Phase 5 with its reason.
- **Georgian-language alias coverage generally.** The class word was genuinely absent from the
  user's question here, so this trace does not evidence a translation gap. Phase 6 adds unit and
  en-dash handling only.
- **`get_end_user_prices` SQL and returned columns.** Unchanged throughout; Phase 6 adds a
  selection parameter only.
