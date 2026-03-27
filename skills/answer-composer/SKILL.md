---
name: answer-composer
description: Use when generating the final user-facing answer from query results and statistics. Governs summary production (Stage 4), including answer structure selection, formatting rules, and grounding requirements. Not for SQL generation or domain reasoning.
---

# Answer Composer

Use this skill when producing the final analytical answer from data results and statistics. This skill enriches the structured summarizer (`llm_summarize_structured`) with the analytical depth previously only available in the legacy fallback (`llm_summarize`).

At runtime, the loader selects references conditionally based on query type and focus:

- [references/formatting-rules.md](references/formatting-rules.md) — always loaded for number formatting and column name translation.
- [references/focus-guidance-catalog.md](references/focus-guidance-catalog.md) — the Always section plus focus-specific section are extracted per query.
- [references/answer-templates.md](references/answer-templates.md) — loaded when query type matches a template (comparison, trend, etc.).
- [references/balancing-analysis-template.md](references/balancing-analysis-template.md) — loaded for balancing-keyword queries.
- [references/forecast-caveats.md](references/forecast-caveats.md) — loaded when forecast keywords are detected.
- [references/grounding-contract.md](references/grounding-contract.md) — reference documentation only; grounding is enforced programmatically in `agent/summarizer.py`.

## Non-negotiable rules

- Every numeric value must cite its source (data_preview or statistics) — no fabricated numbers.
- No hedging language when data is available (no "probably", "possibly", "სავარაუდოდ", "შესაძლოა").
- Never use raw column names in answers (say "balancing price in GEL" not "p_bal_gel").
- Always include physical units (GEL/MWh, USD/MWh, %, thousand MWh, GEL/USD).
- Match answer structure to query type — a factual lookup gets 1-2 sentences, a driver analysis gets a full structured response.
- Answer ONLY what the user asked — don't discuss unrelated topics.

## Primary job

Select the appropriate answer template based on query type, inject focus-specific guidance, format the response with correct units and data citations, and ensure all claims are grounded in the provided data.
