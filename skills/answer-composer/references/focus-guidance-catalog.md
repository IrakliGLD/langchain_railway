# Focus Guidance Catalog (Summarizer)

Declarative guidance blocks keyed by query focus area for the summarizer stage. Each block provides the focus-specific rules for generating the answer.

## Always (unconditional rules)

### Stay focused
1. Answer ONLY what the user asked — don't discuss unrelated topics
2. If query is about CPI/inflation → discuss CPI only (not electricity prices unless comparing affordability)
3. If query is about tariffs → discuss tariffs only (not balancing prices)
4. If query is about generation/quantities → discuss generation only (not prices)
5. If query is about entities/list → provide the list only (no price analysis)
6. Only discuss balancing price if explicitly asked or if query contains balancing price keywords
7. For analytical queries: provide DETAILED, STRUCTURED answers. For simple lookups: 1-2 sentences is sufficient

### Column name prohibition
NEVER use raw database column names in your answer.
- Wrong: "share_hydro increased", "p_bal_gel rose", "tariff_gel changed"
- Correct: "hydro generation share increased", "balancing price in GEL rose", "tariff in GEL changed"

### Seasonal-adjusted trend analysis (when stats_hint contains "SEASONAL-ADJUSTED TREND ANALYSIS")

These pre-calculated statistics are AUTHORITATIVE — use them directly instead of raw data patterns.

1. Use "Overall growth" percentage for multi-year trends
   - DO NOT compare first month to last month directly
   - DO NOT say "doubled" or "tripled" based on raw monthly data
   - USE the calculated CAGR (average annual growth rate)

2. Pay attention to incomplete year warnings
   - If stats say "Last year has only X months" → mention this caveat
   - DO NOT treat incomplete years as full years in trend analysis

3. For trend queries cite:
   - Year range: "From [first_year] to [last_year]"
   - Overall growth: "increased by [overall_growth_pct]%"
   - CAGR: "average annual growth of [cagr]%"
   - Seasonal pattern: "peak in [peak_month], low in [low_month]"

4. Distinguish between:
   - Long-term trend (use CAGR from stats)
   - Seasonal variation (use peak_month/low_month from stats)
   - Recent momentum (use recent_12m_growth if available)

Example:
- Correct: "From 2015 to 2023, demand increased by 25.5% (overall growth), with an average annual growth rate of 3.2% (CAGR). Demand shows strong seasonality, peaking in January (winter) and reaching lows in July (summer)."
- Wrong: "Demand almost doubled from 171k MWh to 313k MWh" (this compares January to August — pure seasonality!)

### Data availability
- Balancing composition (entity share) data is available ONLY from 2020 onwards.
- If shares show NULL or 0 for periods before 2020, this means data was NOT collected — NOT that the share was zero.
- NEVER say "share was 0%" for pre-2020 periods. Instead say "data is not available for this period."

## Focus: Balancing

**Trigger**: query_focus == "balancing" OR keywords: balancing, p_bal, балансовая, ბალანსის

Full balancing analysis guidance is in [balancing-analysis-template.md](balancing-analysis-template.md). Apply the full structured format with:
- Step-by-step data citation with exact numbers
- Composition (Factor 1) + Exchange Rate (Factor 2) + Seasonal (if applicable)
- Correlation citations from stats_hint
- Confidentiality: DO NOT disclose specific PPA/import price estimates
- For entity pricing details, see domain knowledge topics: `balancing_price`, `currency_influence`

## Focus: Tariff

**Trigger**: query_focus == "tariff" OR keywords: tariff, тариф, ტარიფ

- Tariffs follow GNERC-approved cost-plus methodology
- Thermal tariffs include a Guaranteed Capacity Fee (fixed) plus a variable per-MWh cost based on gas price and efficiency
- Gas is priced in USD, so thermal tariffs correlate with the GEL/USD exchange rate (xrate)
- Do NOT apply seasonal logic to tariff analyses
- Focus on annual or multi-year trends explained by regulatory cost-plus principles

## Focus: CPI / Inflation

**Trigger**: query_focus == "cpi"

- Focus on CPI category 'electricity_gas_and_other_fuels' trends
- When comparing to electricity prices (tariff_gel or p_bal_gel), frame as affordability comparison
- Describe CPI trend direction, magnitude, and time periods clearly
- Only discuss electricity prices if user asks for affordability comparison

## Focus: Generation

**Trigger**: query_focus == "generation"

- Focus on quantities (thousand MWh) by technology type or entity
- Describe generation trends, shares, and seasonal patterns
- Summer vs Winter comparison relevant for hydro vs thermal generation
- Only discuss prices if user explicitly asks about price-generation relationships

## Focus: Energy Security

**Trigger**: keywords: energy security, უსაფრთხოება, independence, dependence

For energy security classification details (local vs import-dependent sources), see domain knowledge topic: `balancing_price`.

When analyzing energy security:
- NEVER say "thermal reduces import dependence" or "thermal is local generation"
- ALWAYS clarify "thermal relies on imported gas" when discussing energy security
- Correct: "Winter import dependence includes direct electricity imports AND thermal generation using imported gas"
- Wrong: "Georgia is self-sufficient when using thermal plants"
- Use `tech_quantity_view` for analysis: sum thermal + import as import-dependent; sum hydro + wind + solar as local; calculate local_share = local / (local + import_dependent)
