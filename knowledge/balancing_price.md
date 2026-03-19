# Balancing Price Formation and Drivers

## Definition
The balancing price is the weighted-average price of electricity sold on the BALANCING MARKET (not the general market or exchange), operated by ESCO monthly. ESCO is sole buyer of all balancing electricity. All non-contracted electricity is sold to ESCO as balancing electricity. "renewable_ppa" and "thermal_ppa" mandatorily sell only on balancing segment, they are not allowed to sell with bilateral contract or exchange.

**CRITICAL:** This definition and all analytical logic in this document apply to the CURRENT TRANSITIONAL MARKET MODEL (monthly balancing model with central dispatch).  
For all data analysis, ALWAYS use the current model logic.

**Target Model Note (Brief):**
In the future target model (planned ~July 2027), balancing price will no longer be a monthly weighted average. It will shift to hourly, marginal pricing under self-dispatch.  
However, this document applies ONLY to the current model.

---

**CRITICAL TERMINOLOGY:** when talking about balancing electricity price or balancing electricity, ALWAYS say "balancing market" or "balancing segment" — NEVER shorten to just "market". "market" is general term which includes "balancing market", "bilateral market" and "exchange" segments.

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

---

## Core Pricing Logic (System Level)

**CRITICAL:** In the current transitional model:
- The balancing market is the **core electricity settlement layer**
- A **significant share of total system electricity** is sold as balancing electricity
- Therefore, the balancing price acts as the **reference price for the system**

**Causal Chain:**
Central dispatch (GSE) → determines generation mix → determines balancing composition → determines balancing price (WAP)

There is:
- no marginal pricing
- no hourly clearing
- no merit-order price formation

Hence, price is driven by **composition**, not marginal cost.

**Structural Role of PPA/CfD:**
- CfD and PPA generation form a **core component of balancing supply**
- They act as a **reference base for balancing price formation**

Therefore:
- If lower-cost generation (e.g. deregulated or regulated hydro) is added to a system where CfD/PPA dominates → balancing price decreases  
- If higher-cost sources (e.g. import, regulated_old_tpp) are added → balancing price increases  

This reflects the weighted-average nature of the balancing price.

**IMPORTANT:**  
Not all generated electricity affects balancing price.  
Only electricity **actually sold on the balancing segment** enters price formation.

Therefore:
- High hydro generation does NOT reduce balancing price if that hydro is not sold as balancing electricity  
- Price depends on **balancing composition**, not total generation mix

---

## Data Source
Table: `price_with_usd` — columns: `p_bal_gel` (GEL/MWh), `p_bal_usd` (USD/MWh).  
Coverage: 2020–present, monthly granularity.

---

## Weighting Entities
The following entities participate in the balancing price calculation:

- `deregulated_hydro` — Deregulated hydropower plants, one of the cheapest sources of balancing electricity  
- `import` — Direct electricity imports, USD-priced  
- `regulated_hpp` — Regulated hydro power plants, GEL tariffs, mostly the cheapest source  
- `regulated_new_tpp` — Regulated new thermal power plant (Gardabani), GEL tariff reflecting current xrate  
- `regulated_old_tpp` — Regulated old thermal power plants, GEL tariffs reflecting current xrate  
- `renewable_ppa` — Renewable PPA projects (hydro, solar, wind), USD-priced  
- `thermal_ppa` — Thermal PPA projects, USD-priced  
- `CfD_scheme` — Renewable projects under CfD support schemes, USD-priced  

**Important Structural Note:**
- PPA and CfD electricity are **mandatorily sold on balancing segment**
- They represent a **structural (non-optional) component** of balancing electricity

---

**Calculation Rule:**  
Weights are based on electricity sold as balancing energy by each entity.  
Total balancing quantity = sum of all listed entities from `trade_derived_entities` WHERE `LOWER(REPLACE(segment, ' ', '_')) = 'balancing'`.

---

## Primary Drivers

### 1. Market Composition (Most Important)
**Importance:** PRIMARY DRIVER — Most important for BOTH GEL and USD prices.

Balancing electricity price = weighted average of all categories selling on the BALANCING SEGMENT.

**Reference Role of PPA/CfD in Composition:**
- CfD and PPA volumes are structurally present in balancing electricity
- They act as a baseline level of price in the system

Price changes are driven by deviations from this baseline:
- Adding cheaper sources → lowers average price  
- Adding more expensive sources → increases average price  

- Higher share of cheap sources (`regulated_hpp`, `deregulated_hydro`) → lower price  
- Higher share of expensive sources (`import`, `regulated_old_tpp`, `thermal_ppa`, `renewable_ppa`, `CfD_scheme`) → higher price  

**Seasonal Effect:**
- Summer → high hydro and renewable → lower price
- Winter → balancing mix dominated by PPA/CfD (baseline), regulated thermal, and imports → higher price
  - Higher-cost sources have limited ability to displace this mix because buyers prefer the cheapest available balancing supply

**Balancing vs Total Generation (CRITICAL DISTINCTION):**
- Only electricity sold on the balancing segment affects the balancing price  
- Cheap generation reduces price ONLY if present in balancing electricity mix  
- If cheap hydro is not sold on balancing → it does NOT push price down  

**Structural Evolution:**
- Demand growth increasingly absorbs cheap generation outside balancing  
- As a result:
  - share of cheap hydro in balancing declines  
  - balancing price does not decrease as strongly as in the past, even in high-hydro periods  

**Deregulated Hydro Pricing (CRITICAL):**
- May–Aug: price is low (~<50 GEL/MWh), referenced to cheapest regulated HPP  
- Sep–Apr: price increases (>100 GEL/MWh), referenced to thermal tariffs  
→ In winter months, deregulated hydro becomes **indirectly USD-linked**

---

**Structural Trends:**
- Declining share of `regulated_hpp` and `deregulated_hydro`
- Increasing role of:
  - `renewable_ppa`
  - `import`
  - `thermal_ppa`
  - thermal generation  

→ Structural upward pressure on balancing price

---

**Data Source:**  
`trade_derived_entities` WHERE `LOWER(REPLACE(segment, ' ', '_')) = 'balancing'`

---

**Analysis Requirements:**
- MANDATORY: For long-term trends → analyze by season (summer vs winter)
- MANDATORY: Cite actual share changes
- MANDATORY: Identify cheap vs expensive sources

---

### Transmission Constraints

Even if cheap generation exists (e.g. Enguri HPP), it may NOT affect balancing price if:
- transmission constraints prevent delivery to demand centers

Therefore:
- balancing composition depends on **deliverable energy**, not total generation

---

### Export Effect

If hydro is exported:
- it is NOT available for balancing
- does NOT reduce balancing price

Export acts as competing sink for cheap generation

---

### Exchange Rate
**Importance:** CRITICAL for GEL price, SMALL impact on USD price.

**Variable:** `xrate` (GEL/USD) from `price_with_usd`.

---

**Entity Pricing Structure:**

- **Direct USD-priced:**
  - `renewable_ppa`
  - `thermal_ppa`
  - `import`

- **USD-linked (via gas cost):**
  - `regulated_old_tpp`
  - `regulated_new_tpp`

- **Partially USD-linked:**
  - `deregulated_hydro` (winter months)

- **GEL-priced:**
  - `regulated_hpp`

---

**Mechanism:**
- GEL depreciation → strong increase in GEL balancing price
- Majority of balancing electricity is **directly or indirectly USD-linked**
- See `currency_influence.md` for the full FX transmission logic and the detailed treatment of seasonally USD-linked deregulated hydro

---

**Analysis Requirements:**
- GEL analysis → composition + xrate
- USD analysis → composition only
- USD removes FX noise

---

## Price Hierarchy
From cheapest to most expensive (approximate):

1. Regulated HPP — ~below 50 GEL/MWh  
2. Deregulated Hydro — seasonal (low in summer, high in winter)  
3. Regulated Thermal (old/new TPP) — gas + xrate driven  
4. Renewable PPA — USD-priced (CONFIDENTIAL)  
5. CfD_scheme — USD-priced (CONFIDENTIAL)  
6. Thermal PPA — USD-priced (CONFIDENTIAL)  
7. Import — USD-priced, variable (CONFIDENTIAL)

---

**CONFIDENTIALITY RULE:**  
Do not disclose specific PPA/import prices.

---

## Key Events

- **Jan 2024:** Gas price increase → thermal tariffs increased → higher balancing price  
- **Jul 2024:** Exchange launched (GENEX) — added new trading segment  
- **2020 onwards:** Entity-level data available  

---

## Analysis Guidelines

### For Correlation Analysis
1. Focus on composition  
2. Include xrate for GEL  
3. Use balancing segment data  
4. Support with numeric evidence  

---

### For Price Explanation
1. Start with share changes  
2. Identify drivers (cheap vs expensive)  
3. Add xrate (if GEL)  
4. Separate seasonal vs structural effects  

---

### For Seasonal Analysis
- Summer → hydro/renewable → lower price  
- Winter → thermal/import → higher price  
- Use MoM and YoY comparisons  

---

### Disclosure Rules
- Do NOT disclose PPA/import prices  
- Clarify support schemes = PPA + CfD only  

---

## Price Decomposition

### compute_entity_price_contributions

Calculates monthly entity contributions.

**Output columns:**
- `balancing_price_gel`
- `share_[entity]`
- `price_[entity]`
- `contribution_[entity]`
- `total_known_contributions`
- `residual_contribution_ppa_import`

---

**Available prices:**
- `deregulated_hydro` → `p_dereg_gel`  
- `regulated_hpp` → tariff_with_usd  
- `regulated_new_tpp` → Gardabani  
- `regulated_old_tpp` → Tbilisi, Mtkvari, G-Power  

---

**Unavailable prices:**
- Renewable PPA  
- CfD_scheme  
- Thermal PPA  
- Import  

---

## Analytical Workflow
1. Identify share changes  
2. Estimate contribution impact  
3. Include xrate  
4. Adjust for seasonality  
5. Interpret residual  

---

## Interpretation Guidelines
- Contribution increase → higher price impact  
- Large residual → PPA/import influence  
- Always validate with:
  - seasonality
  - tariffs
  - composition  

