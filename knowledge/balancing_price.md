# Balancing Price Formation and Drivers

## Definition
The balancing price is the weighted-average price of electricity sold on the BALANCING MARKET (not the general market or exchange), operated by ESCO monthly. ESCO is sole buyer of all balancing electricity. All non-contracted electricity is sold to ESCO as balancing elecricity. "renewable_ppa" and "thermal_ppa" mandatorily sell only on balancing segment, they are not allowed to sell with bilateral contract of exchange.

**CRITICAL TERMINOLOGY:** when taling about balancing electricity price or balancing electricity, ALWAYS say "balancing market" or "balancing segment" — NEVER shorten to just "market". "market" is general term which includes "Balancing market", "bilateral market" and "exchange" segments.

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
Coverage: 2020–present, monthly granularity.

## Weighting Entities
The following entities participate in the balancing price calculation:
- `deregulated_hydro` — Deregulated hydropower plants, one of the cheapest source of balancing electricity. 
- `import` — Direct electricity imports, USD-priced
- `regulated_hpp` — Regulated hydro power plants, GEL tariffs. mostly the cheapest source of electricity. 
- `regulated_new_tpp` — Regulated new thermal power plant (Gardabani), GEL tariff reflecting current xrate.  
- `regulated_old_tpp` — Regulated old thermal power plants, GEL tariffs reflecting current xrate.  
- `renewable_ppa` — Renewable PPA projects (hydro, solar, wind) under PPA support schemes, USD-priced
- `thermal_ppa` — Thermal PPA projects, USD-priced
- `CfD_scheme` - Renewable PPA projects (hydro, solar, wind) under CfD support schemes, USD-priced

**Calculation Rule:** Weights are based on electricity sold as balancing energy by each entity. Total balancing quantity = sum of all listed entities from `trade_derived_entities` WHERE `LOWER(REPLACE(segment, ' ', '_')) = 'balancing'`.

## Primary Drivers

### 1. Market Composition (Most Important)
**Importance:** PRIMARY DRIVER — Most important for BOTH GEL and USD prices.

Balancing electricity price = weighted average of all categories selling on the BALANCING SEGMENT.


- Higher share of cheap sources (`regulated_hpp` and `deregulated_hydro` ~below 50 GEL/MWh) → lower price
- Higher share of expensive sources (`import`,`regulated_old_tpp`, `thermal_ppa`, `renewable_ppa`, `CfD_scheme`) → higher price, particularly in compared to summer (may-aug) price. hencce push summer prices up. In winter, `thermal_ppa`, `renewable_ppa`, `CfD_scheme` prices might be lower that `import` and `regulated_old_tpp`.
- Composition changes seasonally: summer = high renewable_ppa and hydro, winter = high thermal and import

**Structural Trends:**
- IMPORTANT: Long-term declining trend in deregulated_hydro and regulated_hpp shares in balancing segment.
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
- USD-priced: renewable_ppa, thermal_ppa, import. While the final tariff is set in GEL/MWH, the tariffs for regulated_old_tpp and regulated_new_tpp directly reflect x-rate as their variable costs almost 100% includes gas price, which is USD-priced.
- partially GEL-priced: deregulated_hydro - in summer it is GEL-prices and reference price is cheapest rehulated_hpp but in winter the reference price is thermal tariff, which is directly affected by usd-priced gas price.
- GEL-priced: regulated_hpp


**Mechanism:**
- When GEL depreciates (xrate increases):
  - GEL price rises significantly (all USD-priced entities convert at higher xrate + GEL-priced entities)
  - Significant share of electricity sold at balancing market are USD-priced, directly or indicrectly. Hence, GEL-priced balancing electricity price is significantly depends on the xrate.

**Analysis Requirements:**
- For GEL price analysis: xrate is a MAJOR factor alongside composition
- For USD price analysis: composition is PRIMARY driver
- When comparing GEL vs USD price trends: USD price shows composition effect with no xrate noise

## Price Hierarchy
From cheapest to most expensive (approximate ranges):
1. Regulated HPP (cheapest) — ~below 50 GEL/MWh
2. Deregulated Hydro — varies,  below 50 GEL/MWh in may-aug period (rference price is the cheapest regulated hpp), more than 100 GEL/MWH in other months (reference is thermal price).
3. Regulated Thermal (old TPP, new TPP/Gardabani) — GEL tariffs reflecting current xrate and gas prices
4. Renewable PPA — USD-priced (CONFIDENTIAL)
5. CfD_scheme — USD-priced (CONFIDENTIAL)
6. Thermal PPA — USD-priced, (CONFIDENTIAL)
7. Import — USD-priced, typically expensive but not always (CONFIDENTIAL)

**CONFIDENTIALITY RULE:** Specific PPA and import price estimates are for INTERNAL ANALYSIS ONLY. Never disclose these numbers to users. Say "market-based" or "varies" when discussing.


## Key Events

- **Jan 2024:** Gas price increase → regulated thermal tariffs raised substantially. Pushed winter balancing price up via thermal component. (See tariffs.md for details.)
- **Jul 2024:** Day-ahead electricity exchange launched (GENEX). New market segment alongside balancing.
- **2020 onwards:** Entity-level balancing composition data first available. Pre-2020 share analysis not possible.

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
- When analyzing balancing electricity price change for a specific month,because of seasonality,always make comparison to the previous month (MoM) and the same month from the previous year (YoY)

### Disclosure Rules
- DO NOT disclose: Specific renewable PPA prices (say CONFIDENTIAL)
- DO NOT disclose: Specific thermal PPA prices (say CONFIDENTIAL)
- DO NOT disclose: Specific import prices (say CONFIDENTIAL)
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
- `deregulated_hydro` — the price at which undontracted electricity generated by deregulated power plants is provided in maerialized view `public.price_with_usd` in column `p_dereg_gel` for GEL/WMH price. Mostly GEL-priced, but in SEP-APR period its price is indexed by theraml power plant tariff, with USD-priced gas consumption.
- `regulated_hpp` — Regulated hydro power plants, GEL tariffs. regulated tariffs are provided in `public.tariff_with_usd` in column `tariff_gel`. All entities from that table, except `tbilsresi tpp`, `mktvari tpp`, `gpower tpp`, `gardabani tpp`, `mktvari tpp` and `tbilsresi tpp` are hydro power plants. 
- `regulated_new_tpp` — Regulated new thermal power plant (Gardabani), GEL tariff reflecting current xrate.  regulated tariffs are provided in `public.tariff_with_usd` in column `tariff_gel`. the entity under name `gardabani tpp` fall in this category.
- `regulated_old_tpp` — Regulated old thermal power plants, GEL tariffs reflecting current xrate.  regulated tariffs are provided in `public.tariff_with_usd` in column `tariff_gel`. the entities under name `tbilsresi tpp`, `mktvari tpp`, `gpower tpp` fall in this category.


**Unavailable prices:**
- Renewable PPA: confidential, not in database
- Cfd_scheme: confidential, not in database
- Thermal PPA: confidential, not in database
- Import: varies by transaction, not in database

### Analytical Workflow
1. Use share_changes to identify which entity shares changed  month-over-month or year-over-year
2. Use entity_price_contributions to estimate the price impact of those share changes
3. Consider xrate changes for GEL-denominated price analysis
4. Link composition shifts to seasonal patterns
5. For entities without price data (PPAs, CfD_scheme, imports), infer direction from residual_contribution

### Interpretation Guidelines
- A positive contribution increase indicates that entity contributed more to the price change
- Compare contribution changes to price changes to assess relative importance
- Large residual_contribution_ppa_import suggests PPAs/imports drove price, but exact decomposition unknown
- Always validate against seasonal patterns and tariff changes

## Balancing Market Logic
- Low-hydro months push balancing to thermal and imports, raising volatility and cost
- Balancing prices reflect the residual mix, not just cost; cheap sources depresses prices, expensive one raise them
- `renewable_ppa` and `CfD_scheme` are the most expensive sources in summer.
- `import` and `regulated_old_tpp`
