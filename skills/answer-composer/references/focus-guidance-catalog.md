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
- Composition (Factor 1) + Source Price / Tariff Layer (Factor 2 when present) + Exchange Rate + Seasonal (if applicable)
- Correlation citations from stats_hint
- Confidentiality: DO NOT disclose specific PPA/import price estimates
- For entity pricing details, see domain knowledge topics: `balancing_price`, `currency_influence`
- When `price_*`, `contribution_*`, or tariff columns are present, cite those exact values instead of giving a composition-only explanation

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

## Focus: Trade

**Trigger**: query_focus == "trade" OR keywords: import, export, trade, იმპორტი, ექსპორტი

- Focus on electricity import/export volumes and cross-border flows
- Georgia imports primarily from Russia and Azerbaijan; exports to Turkey and Armenia
- Import volumes are seasonal — higher in winter when hydro generation drops
- Thermal generation using imported gas is NOT the same as direct electricity import; distinguish clearly
- Use `tech_quantity_view` for import/export quantities when available
- Only discuss prices if user explicitly asks about import/export pricing

## Focus: Energy Security

**Trigger**: keywords: energy security, უსაფრთხოება, independence, dependence

For energy security classification details (local vs import-dependent sources), see domain knowledge topic: `balancing_price`.

When analyzing energy security:
- NEVER say "thermal reduces import dependence" or "thermal is local generation"
- ALWAYS clarify "thermal relies on imported gas" when discussing energy security
- Correct: "Winter import dependence includes direct electricity imports AND thermal generation using imported gas"
- Wrong: "Georgia is self-sufficient when using thermal plants"
- Use `tech_quantity_view` for analysis: sum thermal + import as import-dependent; sum hydro + wind + solar as local; calculate local_share = local / (local + import_dependent)

## Focus: Regulation

**Trigger**: vector-retrieved chunks have document_type in {regulation, law, order}

### Enumeration discipline (apply when the question expects a list)

When the user asks "who can / who is eligible / what are the requirements / what documents / what obligations" and the source enumerates items (numbered, lettered, bulleted, or named):

- Output every distinct item present in the source. Do not merge two source items into one bullet, do not split one source item into two, do not skip items because they look unimportant.
- Do not introduce categories not in the source. If the source does not list "Commercial Importers (Traders)", do not add that line — even if it is a real-world category that exists elsewhere in the market.
- For each item, keep the source's own qualifier and scope. If the source says "registered as X with Y, and meeting condition Z", reproduce all three; do not drop the qualifier.
- If two items in the source belong to the same category but have different conditions, list them separately so the conditions stay attached to the right item.
- If the source enumerates items in a numbered article, include the article reference inline next to each item or in a leading sentence.

#### Worked example (illustrative shape, not a real source)

Source enumerates: "Article 4: The following may participate: (a) producers with installed capacity above 5 MW; (b) suppliers registered with the regulator since 2021; (c) traders licensed under Article 17."

Correct answer: three bullets, each preserving its qualifier — (a) producers with capacity above 5 MW, (b) suppliers registered with the regulator since 2021, (c) traders licensed under Article 17.

Incorrect answer: a single sentence "producers, suppliers, traders, and importers are eligible" — this drops conditions, loses the article reference, and adds "importers" which the source did not list.

### General regulation rules

- Treat EXTERNAL_SOURCE_PASSAGES as the authoritative primary source
- Structure the answer following the source document's own structure:
  - For procedures/registration: numbered steps or sequential requirements
  - For eligibility rules: list ALL conditions; preserve "all of" vs "any of" distinctions
  - For definitions: preserve exact legal terminology, then explain in plain language
- Cite article/section numbers **in the user's response language**, not the source language:
  - English answer → "Article 14", "Section 8.1", "Paragraph 3"
  - Russian answer → "Статья 14", "Раздел 8.1", "Пункт 3"
  - Georgian answer → "მუხლი 14", "მონაკვეთი 8.1", "პუნქტი 3"
  - The vector-retrieved section headers may contain Georgian text like `მუხლი 14. …` because the source documents are Georgian regulations. Translate the label term (`მუხლი` → `Article` / `Статья`) before citing; preserve the article number unchanged.
  - Do NOT copy Georgian section titles verbatim into an English or Russian answer. The article reference is the citation; the Georgian phrase is not.
- Preserve regulatory language from source — do not paraphrase legal terms loosely
- If source passages are incomplete for the requested detail, say so explicitly rather than filling gaps from general knowledge
- Use DOMAIN_KNOWLEDGE as background context only — do not let it override regulatory details from source passages
- Length: adapt to content complexity (100-600 words); regulatory procedures may need more space than definitions
