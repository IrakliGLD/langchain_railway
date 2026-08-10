# Analyzer Prompt Cache Ordering — Implementation Plan

> **For agentic workers:** use `skills/developer-phased-audit`. Do not move to
> the next phase until the current one is planned, implemented, audited
> independently, and its findings fixed.

**Goal:** Make the analyzer's ~8,700 tokens of constant prompt content
cacheable by moving it ahead of the variable content, without changing
Standard's routing until evidence says it is safe.

**Tech Stack:** Python 3.11 (container) / 3.14 (local), pydantic v2, pytest,
`langchain-openai==1.3.5` / `openai==2.46.0`.

**Revision 2 (2026-08-09).** Rewritten after review. Phase 0 is now **executed,
not proposed** — its numbers are below. Six review items are adopted, three are
qualified or rejected with reasons in *Review disposition*.

---

## The issue

A report job spends 64,000–78,000 prompt tokens and caches **1.5–6%**.
Measured per stage across six production jobs:

| Stage | Prompt tokens | `cached_prompt_tokens` |
|---|---|---|
| research planner | 4–6 k | 1,673 on jobs 2+ in a container, else 0 |
| **question analyzer ×4** | **~10–11 k each** | **0, always** |
| analysis writer | 15–23 k | 0, always |
| synthesis writer | 18–27 k | 0, always |
| document repair | 3–17 k | 1,204 |

The analyzer prompt opens with `UNTRUSTED_USER_QUESTION`
(`_ANALYZER_PINNED_HEAD`, `core/llm.py:3115`), and the renderer appends the
output schema *after every block* (`_render_analyzer_prompt`,
`core/llm.py:3331`). So the one variable element leads, the largest constant
trails, and everything constant in between is unreachable by a prefix cache.

Measured across 40 prompt variants (9 query profiles × 4 context combinations,
`scratchpad/probe_header.py`):

```
worst-case pairwise common prefix, today:   28 chars  (~7 tokens)
```

Four analyzer calls per report ⇒ **~40 k tokens per job** uncacheable by
construction. Standard pays the same shape once per query, at higher volume.

This inverts a convention the codebase already states, in `core/llm.py` at the
document writer: *"Constants first: prefix caching only pays off ahead of
per-request content."* The analyzer never got it.

### A second finding, from Phase 0

`ANALYZER_PROMPT_BUDGET_MAX_CHARS` defaults to 45,000 with a 10% safety margin
⇒ **40,500 effective**. Full analyzer prompts measure 39,209–49,581 chars.
**36 of the 40 matrix cases exceed the budget and are truncated today.**

Two consequences. First, the truncation interaction is load-bearing, not an
edge case — any ordering change has to be proven against post-budget text, and
Phase 0 was run that way. Second, the analyzer is shedding catalog content on
almost every call. That is a real problem and it is **not** this plan's
problem; recorded in *Out of scope*.

---

## Phase 0 — executed. Decision gate: **PASS**

Component sizes, current code:

| Component | Chars |
|---|---|
| output schema (`QuestionAnalysis.model_json_schema`) | 15,490 |
| `CONTRACT_RULES` (`_ANALYZER_CORE_RULES`) | 15,835 |
| `CONTRACT_ANSWER_KIND_GUIDE` | 2,130 |
| `CONTRACT_QUERY_TYPE_GUIDE` | 1,258 |
| **guaranteed header, rendered** | **34,886** |
| system message (identical across queries) | 866 |

None of the four header blocks appears in `_ANALYZER_TRUNCATION_DATA` or
`_ANALYZER_TRUNCATION_KNOWLEDGE` (`core/llm.py:3466`, `:3478`), and
`tests/test_question_analyzer_phase_c.py:1718` already asserts that
disjointness. They cannot be dropped.

Simulating the proposed order over the same 40 variants, **after** applying the
real budget:

```
worst-case pairwise common prefix, today:         28 chars  (~7 tokens)
worst-case pairwise common prefix, proposed:  34,886 chars  (~8,721 tokens)
schema intact at head in all 40 cases:        True
question block present in all 40 cases:       True
```

The matrix covered scalar, comparison, threshold, explanation, forecast,
scenario, knowledge, clarify (with history), a realistic report-track composite
and a maximum-length report-track composite; each crossed with
±`TRUSTED_PREVIOUS_CONTRACT` and ±`TRUSTED_EVIDENCE_ANOMALY`. Truncation fired
in 36 of the 40 and the header survived every one, because the header is
family-invariant and section-aware truncation edits eligible section *content*
in place without moving earlier sections.

**8,721 tokens against a 2,000-token stop condition. Gate passes.** Proceed.

---

## The fix

Render the analyzer prompt as:

```
1. output schema            ← raw text, BEFORE the first tagged section
2. CONTRACT_QUERY_TYPE_GUIDE
3. CONTRACT_ANSWER_KIND_GUIDE
4. CONTRACT_RULES
——— everything above is byte-identical across queries ———
5. UNTRUSTED_USER_QUESTION
6. TRUSTED_EVIDENCE_ANOMALY / TRUSTED_PREVIOUS_CONTRACT  (when present)
7. remaining blocks, existing relative order per prompt family
```

One code path for both modes, selected by one env variable. Standard's prompt
is byte-identical to today's while the selector is `off`.

### Why the schema goes in the prefix, not into a tagged block

`_protected_section_fallback_truncate` (`core/prompt_budget.py:155`)
reconstructs the prompt from three parts: text before the first tagged section,
the surviving tagged sections, and text after the last one. **Untagged text
placed between tagged sections is silently discarded.** So the schema cannot
simply be moved "up"; it has to be either the prefix or a tagged block.

Both work — `_SECTION_CONTENT_RE` accepts a `CONTRACT_\w+` tag, so a
`CONTRACT_OUTPUT_SCHEMA` block would be matched and protected. Prefix wins on
one criterion: the schema text stays **byte-identical** to today
(`Respond with JSON exactly matching this schema:\n{…}`), so the only change
the model sees is position. A tagged block would also rewrite the wording, and
three tests pin that sentinel string surviving truncation
(`tests/test_question_analyzer_phase_c.py:1731`, `:1756`, `:1776`).

### Why the stable-header variant, not question-last

An earlier draft put the question absolutely last and reversed the truncation
candidates behind it. Phase 0 says the stable header alone already reaches
8,721 tokens — 86% of the effective budget. The aggressive variant's marginal
gain is small, and it moves the question past every catalog, which is a much
larger recency change. **Take the measured 8,721 tokens; leave the tail
ordering alone.** Revisit only if telemetry says the remainder matters.

### Why one flagged code path, not a report-only prompt

A report-only reorder forks the analyzer prompt into two orderings that drift.
That is the exact failure this session has been removing — the guardrail
reading the wrong text, the label map without its inverse, the gate and
assembler judging length independently. A selector gives the same protection
without the fork.

### Injection safety

Unchanged. The defence is the `<<<…>>>` delimiters plus the system instruction
to treat `UNTRUSTED_*` blocks as untrusted data
(`_analyzer_system_message`, `core/llm.py:3522`), not the block's position.
Labels and delimiters move with the block. The system message already names
`CONTRACT_*` and `RULE_*` as authoritative, which is now *more* consistent with
the layout, not less.

---

## Review disposition

### Adopted

| # | Item | Verified at |
|---|---|---|
| 1 | The schema must move too, or 15,490 chars stay behind the question | `core/llm.py:3331` |
| 2 | Untagged interstitial text is dropped by the emergency fallback ⇒ schema goes in the prefix | `core/prompt_budget.py:155-181` |
| 3 | The question is not the only variation source; Phase 0 must be a matrix over families, context blocks, lengths, and post-budget text | `core/llm.py:3239-3318`, `:3466-3489` |
| 4 | Prefer the stable header; defer question-last | Phase 0 measurement above |
| 5 | Flag cannot be boolean — use `off｜report｜all`, default `off` | — |
| 6 | Prompt-order identity must enter the application response-cache key | `core/llm.py:3583` |
| 7 | Report analyzers run concurrently, so intra-fan-out cache hits are not the thing to validate | `core/report_job_processor.py:667` |
| 8 | "Cold container" was the wrong boundary — the response cache is process-local, provider caching is not | — |
| 9 | Measure cache **writes**, not only reads | `core/llm_runtime.py:77-117` |
| 10 | The 18-case Standard golden does not exercise report-track prompt shape | `agent/report_research_execution.py:903` |

Item 10 is the strongest of the set. Report-track analyzer input is a composite
— first question, then `Research track:`, `Required coverage:` bullets, and
`Report context:` — and that shape has **already** caused four routing misroutes
this quarter, fixed by making positive routing conditions read only the leading
question. A prompt-order change is exactly the kind of edit that could disturb
it again, and the Standard golden would not see it.

### Verified GPT-5.6 cache controls

GPT-5.6 caches exact prefixes at cache breakpoints and does not fall back to the
longest matching unmarked prefix. Because the analyzer was sent as one user
message, its implicit breakpoint included the changing question and report
context. That explains the observed `cached_tokens=0` plus repeated full-prompt
writes despite an 8,721-token common prefix.

The supported fix is an explicit `prompt_cache_breakpoint` after the stable
header, the same `prompt_cache_key` on matching calls, and request-wide
`prompt_cache_options={"mode": "explicit", "ttl": "30m"}`. The pinned
`langchain-openai==1.3.5` preserves the breakpoint on text content blocks and
forwards both request arguments. OpenAI documents these GPT-5.6 semantics in
the [prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching).

GPT-5.6 cache writes cost 1.25× the uncached input rate, so Phase 4 records
reads, writes, latency, and total prompt tokens before making a savings claim.

**Routine gate reduced to `pytest tests/ --ignore=tests/security -q`.** That is
the repo's documented targeted suite, but since 2026-07-19 the targeted and
full suites both run in ~30 s, so dropping `tests/security` buys nothing and
loses coverage. Every phase runs the **full** suite plus `ruff`. The real
distinction the review is reaching for is *live-model* gates — the routing
golden and the redteam gate — and those are scheduled only at the phases that
change prompt text or flip a selector.

---

## Phases

Offline gate, every phase: `python -m pytest tests/ -q` · `ruff check .`
Run from `D:\Enaiapp\langchain_railway` with the standard test env.
Live gate, where noted: `python -m guardrails.redteam_gate` (≥ 0.92) and
`evaluation/routing_golden_set.py`.

**Status at 2026-08-09.** Phases 0–2 done and committed. Phase 3's harness and
Phase 4's code are delivered, but their *runs* are blocked: this environment
has no live model credentials and no deploy access. Everything mechanically
possible is finished; what remains is three commands an operator runs.

| Phase | State | Commit |
|---|---|---|
| 0 Measure | done — 8,721 tokens, gate passed | — |
| 1 Pin legacy | done — 10 tests, mutation-checked | `ccfc2ed` |
| 2 Implement, dark | done — selector defaults `off` | `dfe17f9` |
| 3 Report-track eval | harness delivered, **run blocked** | `2541581` |
| 4 Canary | code delivered, **run blocked** | `df532ea` |
| 5 Standard decision | **blocked** (live golden) | — |
| 6 Optional | not started | — |

### Operator runbook for the blocked phases

```bash
# Phase 3 — needs the production env for the active MODEL_TYPE
python evaluation/analyzer_prompt_order_pairs.py --repeats 3
```

Exit 0 means no stable routing disagreement and no schema-adherence
regression. Investigate any `DIFFER` line individually; `noisy` lines are model
variance, not blockers.

```bash
# Phase 4 — on the worker, only after Phase 3 is clean
ENAI_ANALYZER_CONSTANTS_FIRST=report
ENAI_ANALYZER_EXPLICIT_PROMPT_CACHE=true
```

Run **two reports inside the provider's cache TTL** — not two calls inside one
report's fan-out, which are concurrent and race each other's writes. Read
`cached_prompt_tokens` and `cache_write_tokens` off the
`llm_response_telemetry` line for `stage=report_question_analyzer`. The explicit
cache flag implies the stable schema-derived key; the standalone
`ENAI_ANALYZER_PROMPT_CACHE_KEY` flag may remain off.

```bash
# Phase 5 — after Phase 4 shows the mechanism works
python evaluation/routing_golden_set.py
ENAI_ANALYZER_CONSTANTS_FIRST=all python evaluation/routing_golden_set.py
```

### If schema adherence degrades — the fallback ladder

Constants-first moves the schema and `CONTRACT_RULES`, the two
highest-authority artifacts, out of the prompt's last position. Every
recurring analyzer failure in production has been a schema violation
(`83010f04`, `3b92f462`, `c7823cc9`), so this is the most likely way the
change hurts. A regression does **not** mean reverting:

| Header | Prefix | Rules position |
|---|---:|---|
| schema + guides + `CONTRACT_RULES` | 8,721 tokens | early |
| schema + guides only | **~4,750 tokens** | back behind the question |

Still 2.4× the decision gate. Phase 3 counts sanitizer repairs per arm
precisely so this is decided on evidence rather than on feel.

### A coupling worth knowing

The knowledge topic stems are embedded in the header, via `KnowledgeTopicName`
inside the schema. **Adding a knowledge topic rolls the analyzer prompt
cache** — one cold prefix after each such release, not a per-request cost.
This is why `prompt_cache_key` is derived from a schema digest rather than
hand-versioned: the key then moves automatically exactly when the header does.

### Phase 0 — Measure the achievable prefix ✅ **DONE**

Result above: 8,721 tokens post-budget worst case across 40 variants. Gate
passes. Probes live in the session scratchpad; Phase 1 re-establishes the
measurement as a repository test.

### Phase 1 — Pin what can be pinned

No production behaviour change.

- **Legacy prompt hash.** Snapshot the rendered prompt for a fixed matrix of
  queries and assert byte equality. This is the mandate — "no impact on
  Standard" — expressed as an assertion instead of a promise, and it must exist
  *before* Phase 2 so it can fail.
- **Block-set equivalence.** Assert the reordered prompt contains the same
  block names with the same bodies as the legacy one — set equality on
  name → body, **including the schema text**, which the review correctly noted
  a name→body comparison would otherwise miss.
- **Delimiter and label preservation** for every block.
- **Stable-header purity:** assert the first 34,886 chars are identical for two
  unrelated queries, and that no header block appears in either truncation
  priority list.
- **Truncation survival:** rebuild the Phase 0 matrix as a test — worst-case
  post-budget common prefix ≥ 30,000 chars, and both `_section_aware_truncate`
  and `_protected_section_fallback_truncate` preserve the header intact.
- Record a baseline run of the 18-case routing golden. *(live)*

### Phase 2 — Implement behind `ENAI_ANALYZER_CONSTANTS_FIRST`

Validated selector, not a boolean:

```python
_ANALYZER_ORDER_MODES = ("off", "report", "all")

def _analyzer_prompt_order(report_profile: bool) -> str:
    """'legacy' or 'constants_first' for this call."""
    mode = (os.getenv("ENAI_ANALYZER_CONSTANTS_FIRST", "off") or "off").strip().lower()
    if mode not in _ANALYZER_ORDER_MODES:
        log.warning("Unknown ENAI_ANALYZER_CONSTANTS_FIRST=%r; using 'off'", mode)
        mode = "off"
    if mode == "all" or (mode == "report" and report_profile):
        return "constants_first"
    return "legacy"
```

Resolved once in `llm_analyze_question` and threaded as a keyword into
`_build_analyzer_prompt_blocks` and `_render_analyzer_prompt`, both defaulting
to `"legacy"` so every existing caller and test is untouched.

Also in this phase:

- **Cache identity.** Append `|order={order}` to `cache_input`
  (`core/llm.py:3583`), so a cached analysis produced under one ordering is
  never served under the other. Without this, an A/B is not an A/B.
- **Cache-write telemetry.** Extend `_extract_cached_prompt_tokens`'
  neighbourhood in `core/llm_runtime.py` with a sibling that reads
  `cache_creation` / `cache_creation_input_tokens`, and log both alongside the
  existing read count.
- `tests/test_contract_continuity.py:113` pins
  `names[0] == "UNTRUSTED_USER_QUESTION"` and the previous-contract block
  second. It passes unchanged with the selector off; add the
  constants-first counterpart asserting the previous-contract block still
  directly follows the question.

Default stays `off` on deploy. This phase ships dark.

### Phase 3 — Report-track semantic evaluation

Before any activation. Paired legacy/constants-first analysis over sanitized
real report-track queries — the `Research track:` / `Required coverage:` /
`Report context:` composite shape, not Standard's one-liners.

Compare routing decisions field by field. **Repeat every changed case** before
attributing it to prompt order; the model is nondeterministic and a single-run
difference proves nothing. *(live)*

Nothing activates until this is clean.

### Phase 4 — Report canary

Set `ENAI_ANALYZER_CONSTANTS_FIRST=report` and
`ENAI_ANALYZER_EXPLICIT_PROMPT_CACHE=true` on the worker. Run **at least two
reports inside the 30-minute minimum TTL** — not two calls inside one report's
fan-out, which are concurrent and racing each other's writes.

Record per analyzer call: prompt tokens, cache reads, cache writes, latency.
Confirm identical `stable_prefix_sha256` values, then compute the saving from
the published read/write rates. The first sequential request should write the
stable prefix; a later request with the same hash and key should read it while
leaving the dynamic suffix uncached. If explicit mode still reads 0, capture
the hashes and telemetry for a provider support case.

The stable schema-derived key and explicit options are scoped to the analyzer
stage and allow-listed to OpenAI. Older OpenAI models and all other providers
retain their historical invocation shape. *(implemented dark; provider canary
pending)*

### Phase 5 — The Standard decision

Run the 18-case golden with the selector `off`, then `all`. Diff the routing
decisions and **review each disagreement individually**, not as a pass rate;
re-run changed cases as in Phase 3. Flip to `all` only if there is no
regression. *(live)*

A reorder that changes nothing is the expected outcome; one that *improves*
Standard is plausible, since instructions nearer the end tend to be followed
more closely.

### Phase 6 — Optional

Question-last / reverse-tail ordering, and the writer's ~810-token prefix. Only
if Phase 4 telemetry says the remaining uncached share is worth another
behaviour change.

---

## Risks

- **Thin eval.** 18 live Standard cases plus a hand-built report-track set
  cannot prove no regression across unbounded natural language. **The selector,
  not the eval, is the safety mechanism.**
- **Recency effects.** Moving the question from first to fifth changes what the
  model attends to most. This is a behaviour change, not a refactor.
- **Report-track shape is already fragile.** Four misroutes this quarter traced
  to how that composite is read. Phase 3 exists for this and gates activation.
- **Provider cache reuse remains empirical.** The breakpoint behavior is
  documented and the serialized request is tested, but Phase 4 must still
  confirm provider reads in production.
- **Prompt-cache controls touch a shared invocation path.** Context-local scope
  and the OpenAI allow-list keep other stages and providers untouched.

## Out of scope

- ~~The analyzer is over budget on 36 of 40 profiles.~~ **Withdrawn
  2026-08-10.** That measurement used the 45,000 default. Production sets
  `ANALYZER_PROMPT_BUDGET_MAX_CHARS=71000` (effective 63,900) and the largest
  prompt the report path can build is ~50,000 chars, so **0 of 40 truncate as
  deployed**. There is no analyzer truncation to fix. The constants-first
  prefix measures 34,886 chars at either budget, so nothing else in this plan
  changes: truncation never bounded the prefix — per-family block selection
  does.
- The writer's short prefix (Phase 6, optional).
- The repair loop's writer-computed arithmetic (`18.5296%`, `6.4393217`) —
  a writer behaviour, not a caching one.
- Chart row sampling — examined and deliberately left alone, because the
  sampler protects the min, max and largest-swing rows the writer cites.
