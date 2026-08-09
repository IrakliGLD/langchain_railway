# Analyzer Prompt Cache Ordering — Implementation Plan

> **For agentic workers:** use `skills/developer-phased-audit`. Do not move to
> the next phase until the current one is planned, implemented, audited
> independently, and its findings fixed.

**Goal:** Make the analyzer's ~10,000 tokens of constant prompt content
cacheable by moving it ahead of the one variable thing in the prompt, without
changing Standard's routing until evidence says it is safe.

**Tech Stack:** Python 3.11 (container) / 3.14 (local), pydantic v2, pytest,
OpenAI `/v1/responses` via `gpt-5.6-terra`.

---

## The issue

A report job spends 64,000–78,000 prompt tokens and caches **1.5–6%** of them.
Measured per stage across six production jobs:

| Stage | Prompt tokens | `cached_prompt_tokens` |
|---|---|---|
| research planner | 4–6 k | 1,673 on jobs 2+ in a container, else 0 |
| **question analyzer ×4** | **~10–11 k each** | **0, always** |
| analysis writer | 15–23 k | 0, always |
| synthesis writer | 18–27 k | 0, always |
| document repair | 3–17 k | 1,204 |

The parallel-fan-out explanation given earlier was **incomplete** — it does not
explain the writers, which run sequentially. Measuring the prompts at rest
found the actual cause, and it is different for the two stages.

**Analyzer — the whole win, and the real defect.** Building the same prompt for
two unrelated questions and diffing:

```
system     866 chars — identical            (~216 tokens)
user    40,491 chars — identical prefix = 28 characters
```

Twenty-eight characters. The prompt opens with `UNTRUSTED_USER_QUESTION`, so
the single variable element leads and roughly **10,000 tokens of schema,
catalogs, filter guide and rule blocks sit behind it**, uncacheable by
construction. Four analyzer calls per report ⇒ **~40 k tokens per job** that no
amount of cache warming can reach. Standard pays the same shape once per query,
at much higher volume.

This is the opposite of the convention the codebase already states, in
`core/llm.py` at the document writer: *"Constants first: prefix caching only
pays off ahead of per-request content."* The analyzer never got that treatment.

**Writer — small, and separately explained.** Its constant lead is ~810 tokens
of `prompt` plus 578 of `system`. Observed caches elsewhere are 1,204 and
1,673, so the writer most likely sits just under the provider's minimum. Even
if fixed, the ceiling is 6–9% of a 15–23 k prompt. **Out of scope here.**

**Cold containers.** Job `177e6bb0` cached 0 on every call; its container was
two minutes old. Whatever is built, the first job per container gets nothing.
This caps the realised benefit and is not fixable from the prompt side.

---

## The fix

Reorder the analyzer prompt so never-truncated constants come first and the
untrusted question comes last. **One code path for both modes**, behind a
single env flag, defaulting **on for report** and **off for Standard**.

### Why this shape and not a report-only prompt

A report-only reorder forks the analyzer prompt into two orderings. Every later
edit then has to be reasoned about twice, and they drift. That is the exact
failure this session has been removing — the guardrail reading the wrong text,
the label map without its inverse, the gate and assembler judging length
independently. Creating a fresh instance of it to avoid touching Standard is a
bad trade. A flag gives the same protection without the fork: Standard's prompt
is byte-identical while the flag is off.

### The constraint that shapes the ordering

The analyzer prompt has section-aware truncation
(`_ANALYZER_TRUNCATION_DATA` / `_ANALYZER_TRUNCATION_KNOWLEDGE`) that drops
blocks when over budget. **A droppable block placed early defeats the whole
exercise**: one truncation difference between two calls breaks the shared
prefix at that point. The order must therefore be

1. blocks that are never truncation candidates (schema, non-listed rule blocks),
2. truncation candidates in reverse-priority order — most-likely-dropped last,
3. `UNTRUSTED_USER_QUESTION`, delimited and labelled exactly as today.

`UNTRUSTED_USER_QUESTION` is not in either truncation list, so it can never be
dropped; moving it last does not risk losing it.

### Injection safety

Unchanged. The defence is the `<<<…>>>` delimiters plus the system
instruction to treat the question as untrusted content, not the block's
position. Labels and delimiters move with the block.

---

## What cannot be done here, and why it matters

The charting work was safe because chart-type selection is a deterministic
function over an enumerable domain: 6,400 points frozen exhaustively, then
mutation-tested. **That technique does not transfer.** The analyzer is an LLM
over unbounded natural language; there is no golden to freeze.

The net that exists:

| Asset | Size |
|---|---|
| `evaluation/routing_golden_set.py` + `.json` | **18 cases**, live LLM, runnable before/after |
| `tests/test_routing_regressions.py` | 5 offline tests |
| `guardrails.redteam_gate` | safety, not routing quality |

Eighteen live cases for the pipeline's semantic centre is thin. The flag exists
precisely because the evidence is weaker than it was for charting.

---

## Phases

Gate for every phase: `python -m pytest tests/ -q`, `ruff check .`,
`python -m guardrails.redteam_gate` (≥ 0.92). Env values per the repo's
standard test set.

### Phase 0 — Measure the achievable prefix. **Decision gate.**

No production code. For both prompt profiles:

- enumerate which blocks are truncation-eligible and which are never dropped;
- build the prompt for two unrelated questions under the *proposed* order and
  measure the identical prefix, in tokens;
- repeat with one question long enough to trigger truncation, to confirm the
  prefix survives a truncation difference.

**Stop condition:** if the achievable constant prefix is under ~2,000 tokens,
the win does not justify touching Stage 0.2 — record the number and stop.

### Phase 1 — Pin what can be pinned

- A test that the reordered prompt contains **the same blocks with the same
  content** as today (set equality on block name → body), so the reorder cannot
  silently drop or alter content.
- A test that `UNTRUSTED_USER_QUESTION` is present, delimited, and last.
- Record a baseline run of the 18-case routing golden for later comparison.

### Phase 2 — Implement behind `ENAI_ANALYZER_CONSTANTS_FIRST`

Report profile defaults on; Standard defaults off. A test asserts the Standard
prompt is **byte-identical** to today's when the flag is off — that is the
mandate, expressed as an assertion rather than a promise.

### Phase 3 — Observe reports

One or two runs. Expect `cached_prompt_tokens > 0` on
`report_question_analyzer` for the second and later calls in a warm container.
If it stays 0, Phase 0's model of the provider's caching is wrong and the plan
is re-opened rather than patched.

### Phase 4 — The Standard decision

Run the 18-case golden with the flag off, then on. Diff the routing decisions
and **review each disagreement individually**, not as a pass rate. Flip
Standard only if there is no regression. A reorder that changes nothing is the
expected outcome; a reorder that *improves* Standard is plausible, since
instructions nearer the end tend to be followed more closely.

### Phase 5 — Optional

The writer's ~810-token prefix. Small, report-side, safe; do it only if
Phase 3 shows the mechanism works.

---

## Risks

- **Thin eval.** 18 live cases cannot prove no regression across unbounded
  natural language. The flag, not the eval, is the safety mechanism.
- **Recency effects.** Moving the question from first to last changes what the
  model attends to most. This is a behaviour change, not a refactor.
- **Cold containers** cap realised benefit; the first job per container gains
  nothing regardless.
- **Provider caching rules are assumed, not documented here.** Phase 3 is what
  confirms them; Phase 0's estimate could be wrong in either direction.

## Explicitly out of scope

The writer's short prefix (Phase 5, optional); the repair loop's
writer-computed arithmetic (`18.5296%`, `6.4393217`) which is a writer
behaviour, not a caching one; and the chart row sampling, which was examined
and deliberately left alone because the sampler protects the min, max and
largest-swing rows the writer needs to cite.
