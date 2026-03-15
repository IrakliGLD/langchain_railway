---
name: sql-planner
description: Use when generating analytical plans and PostgreSQL queries for the Georgian energy market database. Governs the 4-step plan+SQL generation process (Stage 1). Not for summarization or answer formatting.
---

# SQL Planner

Use this skill when generating the analytical plan (JSON) and PostgreSQL SELECT query from a user's natural language question about Georgian energy market data.

Read [references/plan-schema.md](references/plan-schema.md) for the JSON plan format and output structure.
Read [references/chart-strategy-rules.md](references/chart-strategy-rules.md) for chart type selection and dimension separation.
Read [references/guidance-catalog.md](references/guidance-catalog.md) for focus-specific analytical guidance keyed by query type.
Read [references/sql-patterns.md](references/sql-patterns.md) for canonical SQL patterns per view.

## Non-negotiable rules

- No INSERT/UPDATE/DELETE/DDL — SELECT only.
- No SQL regression functions (regr_slope, regr_intercept) — Python layer handles forecasting.
- Always use ENGLISH column aliases in SQL, even for Georgian/Russian queries.
- Use only documented materialized views.
- Do NOT add date filters unless user explicitly specifies a time period.
- Always use `LOWER(REPLACE(segment, ' ', '_')) = 'balancing'` for segment filters in trade_derived_entities.
- trade_derived_entities has data ONLY from 2020 onwards. Always add `date >= '2020-01-01'` for share queries.
- NULL shares mean data is NOT available — never interpret NULL as 0%.
- LIMIT 3750 on all queries as a safety guard.

## 4-step process

1. **Analyze Intent** (internal): What is the user asking? What domain concepts, metrics, and time periods?
2. **Chart Strategy** (internal): What dimensions are involved? Never mix different units on the same chart.
3. **Output Plan** (JSON): Extract intent, target, period, chart_strategy, chart_groups.
4. **Output SQL** (raw SELECT): Single correct PostgreSQL query to fulfill the plan.

## Output format

```
{plan_json}
---SQL---
SELECT ...
```
