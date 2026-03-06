# Balancing Price Formation and Drivers

## Definition
The balancing price is the weighted-average price of electricity sold on the BALANCING MARKET (not the general market or exchange), operated by ESCO monthly.

**CRITICAL TERMINOLOGY:** ALWAYS say "balancing market" or "balancing segment" — NEVER shorten to just "market".

### Terminology in All Languages
- **English:** balancing market / balancing electricity / balancing segment
- **Georgian:** საბალანსო ბაზარი / საბალანსო ელექტროენერგია / საბალანსო სეგმენტი
- **Russian:** балансирующий рынок / балансирующая электроэнергия / балансовый сегмент

**RULE:** NEVER omit the word "balancing" when discussing balancing electricity price.

Examples:
- ✅ CORRECT: "საბალანსო ბაზარზე გაყიდული ელექტროენერგიის საშუალო შეწონილი ფასი"
- ❌ WRONG: "ბაზარზე გაყიდული ელექტროენერგიის საშუალო შეწონილი ფასი"
- ✅ CORRECT: "weighted average price of electricity sold on the balancing market"
- ❌ WRONG: "weighted average price of electricity sold on the market"

## Data Source
Table: `price_with_usd` — columns: `p_bal_gel` (GEL/MWh), `p_bal_usd` (USD/MWh).
Coverage: 2006–present, monthly granularity.

## Weighting Entities
The following entities participate in the balancing price calculation:
- `deregulated_hydro` — Deregulated hydropower plants, GEL-priced
- `import` — Direct electricity imports, USD-priced
- `regulated_hpp` — Regulated hydro power plants, GEL tariffs
- `regulated_new_tpp` — Regulated new thermal power plant (Gardabani), GEL tariff reflecting current xrate
- `regulated_old_tpp` — Regulated old thermal power plants (Mtkvari, Tbilisi, G-POWER), GEL tariffs reflecting current xrate
- `renewable_ppa` — Renewable PPA projects (hydro, solar, wind) under support schemes, USD-priced
- `thermal_ppa` — Thermal PPA projects, USD-priced

**Calculation Rule:** Weights are based on electricity sold as balancing energy by each entity. Total balancing quantity = sum of all listed entities from `trade_derived_entities` WHERE `LOWER(REPLACE(segment, ' ', '_')) = 'balancing'`.

## Primary Drivers

### 1. Market Composition (Most Important)
**Importance:** PRIMARY DRIVER — Most important for BOTH GEL and USD prices.

Balancing electricity price = weighted average of all categories selling on the BALANCING SEGMENT.

- Higher share of cheap sources (`regulated_hpp` ~30-40 GEL/MWh, `deregulated_hydro` ~40-50 GEL/MWh) → lower price
- Higher share of expensive sources (`import`, `thermal_ppa`, `renewable_ppa`) → higher price
- Composition changes seasonally: summer = high renewable_ppa and hydro, winter = high thermal_ppa and import

**Structural Trends:**
- IMPORTANT: Long-term declining trend in deregulated_hydro and regulated_hpp shares
- IMPORTANT: Long-term increasing trend in renewable_ppa, import, and thermal_ppa shares
- Main contributors to balancing electricity now: renewable_ppa (biggest in summer), import, thermal_ppa, regulated_old_tpp, regulated_new_tpp

**Data Source:** `trade_derived_entities` WHERE `LOWER(REPLACE(segment, ' ', '_')) = 'balancing'`

**Analysis Requirements:**
- MANDATORY: For long-term trends or yearly analysis → check composition by season (summer vs winter)
- MANDATORY: Cite actual share changes (e.g., "renewable_ppa increased from 25.3% to 32.7%")
- MANDATORY: Explain which categories are cheap vs expensive when explaining price impact

### 2. Exchange Rate
**Importance:** CRITICAL for GEL price, SMALL impact on USD price.

**Variable:** `xrate` (GEL/USD) from `price_with_usd` view.

**Entity Pricing:**
- USD-priced: renewable_ppa, thermal_ppa, import
- GEL-priced: deregulated_hydro, regulated_hpp, regulated_old_tpp, regulated_new_tpp
- Note: regulated_old_tpp and regulated_new_tpp are GEL tariffs that directly reflect current xrate

**Mechanism:**
- When GEL depreciates (xrate increases):
  - GEL price rises significantly (all USD-priced entities convert at higher xrate + GEL-priced entities)
  - USD price rises slightly (only GEL-priced entities affected)
- The impact on USD price is SMALL because GEL-priced entity shares are very small

**Analysis Requirements:**
- For GEL price analysis: xrate is a MAJOR factor alongside composition
- For USD price analysis: xrate has SMALL impact, composition is PRIMARY driver
- When comparing GEL vs USD price trends: USD price shows composition effect with minimal xrate noise

## Price Hierarchy
From cheapest to most expensive (approximate ranges):
1. Regulated HPP (cheapest) — ~30-40 GEL/MWh
2. Deregulated Hydro — varies, typically 40-50 GEL/MWh equivalent
3. Regulated Thermal (old TPP, new TPP/Gardabani) — GEL tariffs reflecting current xrate and gas prices
4. Renewable PPA — USD-priced, market-based (CONFIDENTIAL)
5. Thermal PPA — USD-priced, market-based (CONFIDENTIAL)
6. Import — USD-priced, market-based, typically expensive (CONFIDENTIAL)

**CONFIDENTIALITY RULE:** Specific PPA and import price estimates are for INTERNAL ANALYSIS ONLY. Never disclose these numbers to users. Say "market-based" or "varies" when discussing.

## Support Schemes Clarification
**CRITICAL:** In Georgia, support schemes are PPA and CfD contracts ONLY.
- PPA (Power Purchase Agreements) — for renewable and thermal projects
- CfD (Contracts for Difference) — for new renewable projects from capacity auctions
- Guaranteed capacity payments for old thermals are a separate support mechanism (not a scheme for new plants)
- Regulated tariffs (regulated_hpp, regulated_old_tpp, regulated_new_tpp) are NOT support schemes — they are cost-plus regulated tariffs set by GNERC

## Analysis Guidelines

### For Correlation Analysis
1. Primary focus: composition (shares of each category in balancing electricity)
2. Secondary focus: xrate (exchange rate) for GEL price analysis
3. Calculate shares from `trade_derived_entities` WHERE `LOWER(REPLACE(segment, ' ', '_')) = 'balancing'`
4. Cite correlation coefficients when available
5. Note structural trends

### For Price Explanation
1. Start by analyzing share_* columns for all categories
2. Cite ACTUAL share changes with specific numbers
3. Explain price impact based on cheap vs expensive categories
4. For GEL price, also analyze xrate change
5. For long-term trends, separate summer vs winter composition analysis
6. Always cite at least 2-3 main share changes

### For Seasonal Analysis
- Summer (April-July): Higher renewable_ppa and deregulated_hydro shares → lower prices
- Winter (Aug-March): Higher thermal_ppa and import shares → higher prices
- For multi-year trends: Calculate average shares for summer vs winter separately

### Disclosure Rules
- DO disclose: regulated tariffs (~30-40 GEL/MWh for HPP), correlation coefficients, structural trends
- DO disclose: That regulated TPP tariffs (while in GEL) reflect current xrate
- DO NOT disclose: Specific renewable PPA prices (say "market-based")
- DO NOT disclose: Specific thermal PPA prices (say "market-based")
- DO NOT disclose: Specific import prices (say "market-based")
- DO clarify: Support schemes = PPA + CfD only (NOT regulated tariffs)

## Price Decomposition

### compute_entity_price_contributions
Calculates monthly entity contributions to balancing price using available reference prices.

**Output columns:**
- `balancing_price_gel`: actual weighted average balancing price
- `share_[entity]`: quantity share of each entity in balancing electricity
- `price_[entity]`: reference price for entity (from tariff_with_usd or price_with_usd)
- `contribution_[entity]`: estimated contribution = share × reference_price
- `total_known_contributions`: sum of all calculable contributions
- `residual_contribution_ppa_import`: unexplained portion (entities without price data)

**Available prices:**
- Regulated HPP: average tariff from main hydro plants (Enguri, Vardnili, Energo-Pro)
- Deregulated hydro: p_dereg_gel from price_with_usd
- Regulated new TPP: Gardabani tariff_gel from tariff_with_usd
- Regulated old TPPs: average of Mtkvari, Tbilisi, G-Power tariffs

**Unavailable prices:**
- Renewable PPA: confidential, not in database
- Thermal PPA: confidential, not in database
- Import: varies by transaction, not in database

### Analytical Workflow
1. Use share_changes to identify which entity shares changed significantly month-over-month
2. Use entity_price_contributions to estimate the price impact of those share changes
3. Consider xrate changes for GEL-denominated price analysis
4. Link composition shifts to seasonal patterns
5. For entities without price data (PPAs, imports), infer direction from residual_contribution

### Interpretation Guidelines
- A positive contribution increase indicates that entity contributed more to raising the price
- Compare contribution changes to price changes to assess relative importance
- Large residual_contribution_ppa_import suggests PPAs/imports drove price, but exact decomposition unknown
- Always validate against seasonal patterns and tariff changes

## Tariff Transmission Mechanism
- Thermal plant tariffs include fixed (capacity) and variable (gas-linked) components — gas price hikes pass through immediately
- Enguri and Vardnili now recover full cost via higher tariff per sold MWh due to Abkhazia supply adjustment (2025)
- Renewable PPAs are USD-indexed and form a price floor for summer market prices
- Thermal tariff increases (Mtkvari, Tbilisi, G-Power) transmit almost directly to winter balancing prices

## Balancing Market Logic
- Low-hydro months push balancing to thermal and imports, raising volatility and cost
- Balancing prices reflect the residual mix, not just cost; cheap hydro depresses prices, gas/import raise them
- Rising renewable PPA share lifts summer prices as it displaces cheap hydro from balancing volumes
