# Generation Mix and Energy Security

## 1. Generation by Technology

### Data Source
Materialized view: `tech_quantity_view`

- Columns:
  - `type_tech` — generation type
  - `quantity` — thousand MWh
  - `time_month`

### Technologies Included
- hydro - generation
- thermal - generation
- wind - generation
- solar - generation
- import - import
- export - export
- supply-distribution -demand
- direct customers - demand
- losses - demand
- abkhazeti - demand

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

---

## 4. Energy Security Analysis

**CRITICAL FACT:**  
Thermal generation uses imported natural gas and cannot be considered fully domestic/local generation.

---

### 4.1 Correct Classification

#### Local Generation (NO import dependence)
- Hydro (regulated HPP, deregulated hydro, reservoir, run-of-river)
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