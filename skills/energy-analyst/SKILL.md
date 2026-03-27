---
name: energy-analyst
description: Use when the system must reason about Georgian energy market concepts, drivers, entity relationships, pricing mechanisms, or seasonal patterns. Provides the analytical framework that the planner and summarizer rely on. Not for answer formatting — only for domain interpretation.
---

# Energy Analyst

Use this skill when interpreting Georgian energy market data, generating analytical reasoning, or deciding which domain concepts are relevant to a user query.

At runtime, `seasonal-rules.md` and `entity-taxonomy.md` are injected into the structured summarizer for energy-domain queries (balancing, generation, trade, energy_security). The remaining references are authoring aids and background context.

Read references conditionally:
- [references/entity-taxonomy.md](references/entity-taxonomy.md) — when the query involves entity categories, pricing sources, or balancing composition.
- [references/seasonal-rules.md](references/seasonal-rules.md) — when the query spans more than 6 months or involves seasonal patterns.
- [references/driver-framework.md](references/driver-framework.md) — when analyzing price drivers or causal factors.
- [references/confidentiality-rules.md](references/confidentiality-rules.md) — when the response might touch PPA or import pricing.
- [references/domain-focus-index.md](references/domain-focus-index.md) — when mapping a query to applicable domain rules.

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
