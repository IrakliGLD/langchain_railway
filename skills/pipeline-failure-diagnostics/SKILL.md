---
name: pipeline-failure-diagnostics
description: Use when a user reports that a real query produced a wrong, incomplete, or absurd answer, when a /ask request crashed or timed out, or when latency or grounding has regressed. Guides developer-side diagnosis of the langchain_railway pipeline: read TRACE logs, classify the failure (schema crash, routing miss, cross-check override, grounding gate, prompt-budget truncation, slow LLM stage), map the symptom to the right stage from docs/active/query_pipeline_architecture.md §3.4, and propose general (not question-specific) fixes that respect the runtime skills (answer-composer, energy-analyst, question-analyzer, sql-planner). Not loaded into LLM prompts at runtime.
---

# Pipeline Failure Diagnostics

Use this skill when triaging a real failure of the `/ask` pipeline — wrong answer, missing items, hallucinated numbers, schema crash, 504 timeout, latency regression, grounding gate failure.

This is a **developer-side** skill. It is intentionally NOT registered in [skills/loader.py](../loader.py) `_EXPECTED_FILES` and NOT injected into any LLM prompt. It guides the human developer (or a future Claude session helping the developer) toward a correct, general fix.

Read the references in this order when diagnosing:

1. [references/log-reading.md](references/log-reading.md) — first, before anything else, when a log is in front of you.
2. [references/failure-taxonomy.md](references/failure-taxonomy.md) — to classify what you see.
3. [references/fix-principles.md](references/fix-principles.md) — before proposing or accepting a code change.

## Core principles

- **§3.4 is the contract.** The Ideal Decision Tree at [docs/active/query_pipeline_architecture.md §3.4](../../docs/active/query_pipeline_architecture.md) says Stage 0.2 (the question analyzer) is the one LLM call that emits the full answer contract; everything downstream just executes. When a stage downstream "re-interprets" the query, that is itself a bug, not a feature. Most quality regressions trace to a downstream re-interpretation overriding what 0.2 already decided.

- **Generalize, don't patch.** A fix should make the next *similar* question better, not just the reported one. If a proposed change would only help the reported query (e.g. a hardcoded keyword), reject it and look for the structural cause one layer up.

- **Prompt vs code vs skill — pick the right layer.** `core/llm.py` analyzer prompt = how the model interprets the question. `agent/pipeline.py` cross-checks = how code validates/overrides. `skills/answer-composer/` = how the rendered answer is shaped. Putting a fix in the wrong layer creates drift between layers; this is one of the most common audit findings.

- **Observed != causal.** Use the same observational language the runtime is held to (see [energy-analyst/SKILL.md](../energy-analyst/SKILL.md)): "the log shows X correlated with Y" rather than "X caused Y" unless you've actually proved it. Apply this to your own diagnoses.

- **Disagreement cases > pass cases.** If the cross-check at `agent/pipeline.py:_cross_check_answer_kind` logged a disagreement, that is the most informative line in the log. Read it before anything else.

## When the answer is wrong but no error appears

Stage 4 narrative output that paraphrases, drops, or hallucinates items is the hardest class to diagnose because nothing crashed. The pattern is almost always one of:

1. The router emitted the wrong `answer_kind` for the question shape.
2. The cross-check overrode the right `answer_kind` to a "safer" one.
3. The wrong template was selected because `query_type` was off.
4. The prompt budget truncated the section that contained the answer.

Always inspect the TRACE log for `stage_0_2_question_analyzer` (`type=`, `answer_kind=`, `confidence=`), then for any `cross-check disagreement` warning, then the `Prompt budget applied` warning.

## What this skill does NOT cover

- SQL correctness debugging (use `sql-planner` skill conventions).
- Vector/knowledge index quality (separate concern).
- Infrastructure failures (504s from network, container restarts, DB pool exhaustion).
- LLM provider outages — read the timestamp pattern: an outage hits all requests, a routing bug hits one query family.
