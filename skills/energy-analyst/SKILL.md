---
name: energy-analyst
description: Use when the system must reason about Georgian energy market concepts, drivers, entity relationships, pricing mechanisms, or seasonal patterns. Provides the analytical framework that the planner and summarizer rely on. Not for answer formatting — only for domain interpretation.
---

# Energy Analyst

Use this skill when interpreting Georgian energy market data, generating analytical reasoning, or deciding which domain concepts are relevant to a user query. This skill is consumed by both the sql-planner (Stage 1) and the answer-composer (Stage 4).

Read [references/entity-taxonomy.md](references/entity-taxonomy.md) for the 7 balancing entity categories and their pricing.
Read [references/driver-framework.md](references/driver-framework.md) for the driver priority hierarchy.
Read [references/seasonal-rules.md](references/seasonal-rules.md) for seasonal analysis rules and CAGR citation.
Read [references/confidentiality-rules.md](references/confidentiality-rules.md) for disclosure boundaries.
Read [references/domain-focus-index.md](references/domain-focus-index.md) to map query focus areas to applicable rules.

## Non-negotiable rules

- Use observational language ("data shows", "associated with", "consistent with") not causal ("X caused Y") unless correlation data backs the claim.
- Mandatory seasonal separation (summer Apr-Jul / winter Aug-Mar) for any analysis spanning more than 6 months.
- Never disclose PPA or import price estimates — say "market-based" or "expensive".
- Always say "balancing market" or "balancing electricity" — never shorten to just "market".
- Always include physical units (GEL/MWh, USD/MWh, %, thousand MWh) — never currency alone.

## Primary job

Provide the domain reasoning framework so that:

- the planner understands what data to retrieve for a given intent,
- the summarizer knows which drivers to cite, in what order, with what evidence,
- both stages use consistent terminology, entity names, and pricing categories.
