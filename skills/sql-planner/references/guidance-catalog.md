# Guidance Catalog

Declarative guidance blocks keyed by query focus area. Replaces the conditional if/else chain in the planner prompt assembly.

## Always (unconditional rules)

- Use ONLY documented materialized views.
- Aggregation default = monthly. For energy_balance_long_mv, use yearly.
- When USD values appear, `*_usd = *_gel / xrate`.
- CRITICAL: trade_derived_entities has data ONLY from 2020 onwards. For balancing composition (share) queries, always add: `date >= '2020-01-01'`. NULL shares mean data is NOT available — never interpret NULL as 0%.

### Date filtering rules

- DO NOT add date filters unless user explicitly specifies a time period.
- If user asks for "trends", "changes over time", "historical" → show ALL available data.
- Only add WHERE date filters if user says: specific year, specific month, "recent N years", "last N months", date range.

Examples:
- "Show balancing price trend" → No date filter
- "What changed in the last 5 years?" → `WHERE date >= CURRENT_DATE - INTERVAL '5 years'`
- "Price in 2024" → `WHERE EXTRACT(YEAR FROM date) = 2024`

## Focus: Balancing

**Trigger keywords** (EN/KA/RU): balancing, p_bal, საბალანსო, баланс

For entity definitions and pricing mechanisms, see domain knowledge topics: `balancing_price`, `currency_influence`.

**Guidance**:
- Weighted-average balancing price = weighted by total balancing-market quantities
- PRIMARY DRIVER #1: xrate (exchange rate) — most important for GEL/MWh price. Use xrate from price_with_usd view.
- PRIMARY DRIVER #2: Composition (shares) — critical for both GEL and USD prices. Calculate shares from trade_derived_entities. Use `LOWER(REPLACE(segment, ' ', '_')) = 'balancing'` for segment filter. Use share CTE pattern, no raw quantities.
- Higher cheap source shares → lower prices; higher expensive source shares → higher prices

## Focus: Seasonal

**Trigger keywords** (EN/KA/RU): season, summer, winter, сезон, ზაფხულ, ზამთარ

**Guidance**:
- Season is a derived dimension: `CASE WHEN EXTRACT(MONTH FROM date) IN (4,5,6,7) THEN 'summer' ELSE 'winter' END AS season`

## Focus: Seasonal-Forecast

**Trigger keywords**: (seasonal keywords) + (trend, ტრენდი, forecast, პროგნოზი, predict, პროგნოზირება, future, მომავალი)

**Guidance** (overrides basic seasonal):
- For seasonal forecast/trend queries, return MONTHLY data WITH a season column
- DO NOT aggregate by season (no GROUP BY season) — this loses time series data
- Pattern: `SELECT month, value, CASE WHEN EXTRACT(MONTH FROM date) IN (4,5,6,7) THEN 'summer' ELSE 'winter' END AS season`
- The Python layer will calculate separate trendlines for summer/winter months
- Example: "forecast winter and summer prices to 2032" → return monthly price data with season column, NOT aggregated seasonal averages

## Focus: Tariff

**Trigger keywords** (EN/KA/RU): tariff, ტარიფი, тариф

**Guidance**:
- Key entities: Enguri (`'ltd "engurhesi"1'`), Gardabani TPP (`'ltd "gardabni thermal power plant"'`)
- Thermal tariffs depend on gas price (USD) → correlated with xrate
- Use tariff_with_usd view for tariff queries

## Focus: CPI

**Trigger keywords** (EN/KA/RU): cpi, inflation, ინფლაცია

**Guidance**:
- CPI data: use monthly_cpi_mv, filter by `cpi_type = 'electricity_gas_and_other_fuels'`

## Focus: Support Schemes

**Trigger keywords** (EN/KA/RU): support scheme, წახალისების სქემა, схема поддержки, ppa, cfd, capacity

For support scheme definitions and mechanics, see domain knowledge topic: `cfd_ppa`.

**Guidance**:
- Regulated tariffs (HPP, old/new TPP) are NOT support schemes — they are cost-plus regulation
- See knowledge for PPA vs CfD distinction and guaranteed capacity mechanics
