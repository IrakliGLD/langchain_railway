# Fix principles

How to choose a fix once you've classified the failure. These principles align with [docs/active/query_pipeline_architecture.md §3.4](../../../docs/active/query_pipeline_architecture.md) and the user's documented preference for general (not question-specific) solutions.

## The §3.4 lens — where does the fix belong?

Stage 0.2 (analyzer) is the one LLM call that emits the answer contract; everything downstream just executes. Use this to locate the right layer for any fix:

| What's wrong | Right layer | Wrong layer |
|---|---|---|
| LLM picks wrong `query_type` / `answer_kind` | analyzer prompt: few-shot examples or rule clarification in `core/llm.py` | `agent/pipeline.py` keyword override |
| Code overrides a correct LLM contract | `agent/pipeline.py` cross-check policy | analyzer prompt |
| Right contract, wrong rendering | runtime skill: `answer-composer` template / `focus-guidance-catalog` | analyzer prompt |
| Right rendering, missing domain rule | runtime skill: `energy-analyst` reference | answer-composer |
| Right answer, hallucinated numbers | upstream evidence enrichment in `agent/analyzer.py` | provenance gate threshold |
| LLM emits invalid enum | analyzer prompt: explicit enum + vocabulary distinction | Pydantic-permissive validator |
| Stage too slow | model choice or prompt budget | retry count |

If a proposed fix lives in the "wrong layer" column, redirect it.

## Generality test

Before accepting a fix, ask: "Will this make the next *similar* question better?"

- A few-shot example covering "trend and structure of [X]" generalizes to "trend and composition of [X]", "evolution and breakdown of [X]". ✅
- A regex match for `"trend and structure of power supply"` does not generalize. ❌
- A cross-check rule gated on `query_type ∈ {regulatory_procedure, conceptual_definition}` generalizes to all legal-list questions. ✅
- A code branch like `if "transitory" in query: answer_kind = LIST` does not generalize. ❌

If you can't write down a class of similar questions the fix would also help, the fix is too narrow.

## Layer-drift audit

After any fix that touches a contract or vocabulary, audit for drift across:

1. **Pydantic enums** in `contracts/question_analysis.py`
2. **Prompt enum lists** in `core/llm.py::_ANALYZER_CORE_RULES`
3. **Catalog descriptions** in `contracts/question_analysis_catalogs.py`
4. **Loader template lookup tables** in `skills/loader.py::_QUERY_TYPE_TO_TEMPLATE_SECTION`
5. **Answer template content** in `skills/answer-composer/references/answer-templates.md`
6. **Cross-check derivation** in `agent/pipeline.py::_QUERY_TYPE_TO_ANSWER_KIND`

These six places must agree. Drift between any two is a latent bug.

## Shadow mode for routing changes

For changes that affect routing (cross-check policy, query_type→answer_kind mapping, path selection), prefer shadow mode before cutover when the existing behavior has been stable:

- Add the new logic but log "would-fire" without applying.
- Run for a sample of real traffic.
- Inspect the disagreement cases (not just the agreeing ones).
- Cut over only when the new logic disagrees with current behavior in a way that is clearly an improvement.

For low-risk narrow gates (e.g. a confidence-thresholded override), direct cutover is acceptable. State the rationale either way.

## Don't fix what isn't broken

- A `WARNING` in the log is not always a fix target. The cross-check `WARNING` is informational; it indicates disagreement, not failure.
- A 35 s response on a complex analytical question is not a regression if the user accepts the answer.
- Don't tune `thinking_budget` to chase a latency hypothesis without evidence the budget is the cause (Pattern G in [failure-taxonomy.md](failure-taxonomy.md) is the canonical example of this trap).

## When a fix needs a runtime-skill change

The runtime skills ([answer-composer](../../answer-composer/), [energy-analyst](../../energy-analyst/), [question-analyzer](../../question-analyzer/), [sql-planner](../../sql-planner/)) are loaded into LLM prompts at request time. A change there affects the model's behavior on every matching query.

Before editing a runtime skill:

1. Confirm the issue is rendering / interpretation, not contract / routing.
2. Check the skill's existing references — the rule may already exist but be buried.
3. Add the rule to the most narrowly-loaded reference file possible, not to the always-loaded `SKILL.md` (cost: every request pays the prompt budget).
4. If the rule applies to a single query family, gate it in `skills/loader.py` rather than always-loading.

## When a fix needs a code change vs a prompt change

| Symptom | Prefer prompt | Prefer code |
|---|---|---|
| Model picks the wrong field value | prompt | — |
| Model is consistent but the contract is wrong | — | code (cross-check) |
| Model is right and code overrides it | — | code (loosen the override) |
| Model emits invalid enum | prompt | — |
| Output format has wrong structure | prompt (skill) | — |
| Threshold needs tuning | — | code (config) |
| New deterministic computation needed | — | code (analyzer enrich) |

When both layers could be touched, prefer the prompt change first — it's lower-blast-radius — and use the code change only if the prompt change doesn't hold.

## Verification before declaring done

For each fix, define what the success log would look like *before* you re-run the query. Examples:

- "After Phase 1, the analyzer for 'why balancing price changed' should produce `answer_kind=explanation` with no schema crash, `analyzer_available=true`, `summary_source=structured_summary`, and gate_passed=true."
- "After Phase 2, the analyzer for 'who can trade…' should still emit `answer_kind=list`, and the cross-check log line should NOT appear (or should appear with `chosen=list`)."

Then run the query and compare. If the log doesn't match the prediction, the fix isn't done — even if the user-facing answer "looks better."

## Don't conflate observed correlation with proven causation

When proposing a fix from log analysis: a single failing trace is correlative, not causal. Multiple traces showing the same pattern raise confidence. A reproducer (same query → same failure on demand) is the only thing approaching causal proof.

Mirror the same observational language the runtime is held to ([energy-analyst/SKILL.md](../../energy-analyst/SKILL.md)): write "the log shows X is associated with Y" rather than "X causes Y", until you've actually proved it.
