# Generation Mix and Energy Security

## 1. Generation by Technology

### Data Source
Materialized view: `tech_quantity_view`

- Columns:
  - `type_tech` — generation type
  - `quantity` — thousand MWh
  - `time_month`

### Categories in `tech_quantity_view`

The materialized view stores both physical-supply categories (generation, import/export) and demand categories under one `type_tech` column. **Generation** and **demand** entries must NOT be summed together; the categories below are split by role.

**Generation (physical supply):**
- hydro
- thermal
- wind
- solar

**Cross-border flows:**
- import (supply-side, enters the system)
- export (demand-side, leaves the system)

**Demand categories (NOT generation):**
- supply-distribution
- direct customers
- losses
- abkhazeti

### Hydropower types matter:

- Reservoir HPP (Enguri):
  - provides stable supply
  - can shift generation

- Seasonal / run-of-river HPP:
  - highly dependent on water inflow
  - drives volatility

### Renewable Integration Constraints:

Renewable expansion depends on:
- availability of flexible capacity (CCGT, reservoir HPP)
- transmission capacity
- system balancing capability
---

## 2. Core Aggregations

### Total Demand
Total electricity demand is calculated as:

- abkhazeti  
- + supply-distribution  
- + direct customers  
- + losses  
- + export  

---

### Total Domestic Generation
Total domestic generation is calculated as:

- + hydro  
- + thermal  
- + wind  
- + solar  

---

## 3. Generation by Ownership (Reference)

Generation can also be analyzed by ownership structure.

### Data Source
Materialized view: `trade_by_ownership`

- Columns:
  - `date`
  - `ownership`
  - `quantity`

### Ownership Groups
- state
- energo-pro group
- vartsikhe 2005 jsc
- inter-rao
- GIG
- georgian water and power jcs
- other (aggregated)

### Usage Note
- Ownership-based analysis is useful for:
  - market concentration assessment
  - dependency on specific companies/groups
  - linking generation structure with tariff and support schemes
- Ownership values are stored exactly as listed above. **`GIG` is uppercase**; every other
  value is lowercase. Equality filters are case-sensitive.

### Concentration Metrics
Materialized view: `ownership_concentration` — one row per month.

- `hhi` — Herfindahl-Hirschman index of generation ownership
- `owner_count` — number of distinct owners generating that month
- `top1_share`, `top3_share`, `top5_share` — share of generation held by the largest N owners
- `total_generation` — the denominator those shares are computed against

Use this view directly for concentration questions rather than recomputing shares from
`trade_by_ownership`; the published values are the authoritative ones.

---

## 3b. Fleet Structure (Capacity Bands, Age, Capacity Factor)

Generation can also be cut by plant size, plant age, and utilisation.

| View | Cut | Columns |
| ---- | --- | ------- |
| `by_capacity` | installed-capacity band | `date, entity, segment, quantity, facility_count` |
| `by_commissioning` | commissioning cohort | `date, entity, segment, quantity` |
| `capacity_factor` | technology × capacity band | `date, technology, capacity_category, capacity_category_order, segment, facility_count, generation_mwh, installed_capacity_mw, hours_in_month, capacity_factor, capacity_factor_percent` |

### Vocabularies
- **Capacity bands (MW)** — `by_capacity.entity` and `capacity_factor.capacity_category` use the
  *same* eight values: `<=5`, `6-10`, `11-20`, `21-50`, `51-100`, `101-200`, `201-500`,
  `more than 500`.
- **Commissioning cohorts** — `by_commissioning.entity`: `<=1990`, `1991-2000`, `2001-2010`,
  `2011-2020`, `after 2020`.
- **Technologies** — `capacity_factor.technology`: `hpp` (hydro), `tpp` (thermal), `wpp` (wind),
  `solar`.

### Usage Rules
- **`segment` currently holds only `'total'`** in all four fleet views and in
  `ownership_concentration`. Do not apply the `trade_derived_entities` `'balancing'` filter
  here — it matches nothing and silently returns an empty result.
- **`capacity_factor` and `capacity_factor_percent` are the same quantity at two scales:** the
  first is a ratio in 0–1, the second is that value ×100. Choose one. Multiplying
  `capacity_factor_percent` by 100 again produces a value 100× too large.
- **Technology × capacity band is sparse.** Not every pair exists in every month (there is no
  500 MW solar). Do not assume a complete grid, and do not read a missing pair as zero.
- `by_capacity` carries both `quantity` (generation) and `facility_count` (number of plants).
  These answer different questions — "how much do large plants generate" versus "how many large
  plants are there" — and `facility_count` is a stock, so it should not be summed across months.

---

## 4. Energy Security Analysis

**CRITICAL FACT:**  
Thermal generation uses imported natural gas and cannot be considered fully domestic/local generation.

---

### 4.1 Correct Classification

#### Local Generation (NO import dependence)
- Hydro (HPP, reservoir, run-of-river)
- Wind (renewable, no fuel imports)
- Solar (renewable, no fuel imports)

---
#### Energy Security (Extended):

Energy security is not only about local generation,
but also about:
- flexibility (storage, reservoir hydro)
- system balancing capability
- transmission reliability

---

#### Import-Dependent Generation
- Thermal (uses imported natural gas)
- Direct electricity import

**Note:**  
Both depend on cross-border energy supply (fuel or electricity).

---

## 5. Analytical Implications

- Thermal generation is **not a substitute for imports** — it is import-dependent
- The real choice for Georgia is:
  - import electricity  
  - OR import gas to generate electricity  

- True energy independence comes from:
  - hydro
  - wind
  - solar

- Winter import dependence includes:
  - direct electricity imports
  - thermal generation using imported gas

- Summer surplus is:
  - based on hydro generation
  - not dependent on imported fuel

---

## 6. Example Statements

- ✅ CORRECT:  
  "Georgia's energy security depends on local renewables (hydro, wind, solar). Thermal generation, while domestic, relies on imported gas and does not reduce import dependence."

- ✅ CORRECT:  
  "In winter, Georgia is import-dependent: direct electricity imports plus thermal generation using imported gas."

- ❌ WRONG:  
  "Thermal generation is local production that reduces import dependence."

- ❌ WRONG:  
  "Georgia can achieve energy independence by increasing thermal capacity."

---

## 7. Energy Balance (Reference)

### Data Source
Materialized view: `energy_balance_long_mv` (GEOSTAT)

### Usage Notes
- Use **yearly aggregation** (not monthly)
- Contains:
  - national energy balances
  - sectoral demand indicators

---

## 8. Analytical Notes

- Always distinguish between:
  - **generation mix (technical)**  
  - **energy security (dependency-based)**  

- Combine this document with:
  - **Currency Influence** → for FX exposure of generation types  
  - **Tariff Structure** → for regulated cost-based components  
  - **Support Schemes (CfD/PPA)** → for contract-based generation  

- Generation mix should be interpreted together with:
  - seasonality (hydro vs thermal)
  - exchange rate (impact on thermal and imports)
  - support schemes (impact on balancing and price formation)