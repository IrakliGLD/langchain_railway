# Currency Influence

## 1. Core Principle

Electricity price analysis in Georgia must explicitly account for GEL/USD exchange rate (`xrate`) effects.

Prices observed in GEL may change because of:
- underlying cost changes
- generation composition changes
- exchange rate movements

For many electricity sources, exchange rate is a major driver because key costs or contract prices are USD-linked.

---

## 2. Main Channels of Currency Influence

### 2.1 Direct USD-Linked Components
The following components are directly USD-priced or USD-indexed:
- Regulated Thermal
- electricity imports
- renewable PPA 
- thermal PPA 
- CfD-supported generation 

These components strongly affect GEL-denominated electricity prices when `xrate` changes.

---

### 2.2 Tariff-Based FX Transmission

Not all tariffs react to exchange rate in the same way.

#### Regulated Thermal Tariffs
- Regulated thermal tariffs are recorded in GEL
- However, they are economically **USD-indexed / USD-driven**
- This is because the main cost component is natural gas, which is USD-linked
- Variable tariff is recalculated based on actual monthly cost and generation
- Therefore, regulated thermal tariffs reflect current `xrate` almost immediately

#### Regulated Hydro Tariffs
- Regulated hydro tariffs are set in GEL for the regulatory period
- They may include some USD-linked costs
- However, FX impact is not reflected immediately
- Exchange-rate effect is reflected only in the next regulatory period

#### Deregulated Hydro Prices
- Deregulated hydro is GEL-priced in observed market outcomes
- Its price is not directly contract-indexed to USD
- FX effect is indirect, through system composition and competition with USD-linked sources

**Balancing Market Reference Mechanism:**
- When electricity from deregulated hydro is sold on the balancing market, its remuneration is linked to reference generation sources:
  - **May–August (summer months):**
    - price is referenced to the cheapest hydro generation
    - weaker direct link to USD
  - **Other months (winter period):**
    - price is referenced to thermal generation
    - thermal generation is gas-based and USD-linked
    - therefore, deregulated hydro becomes **effectively USD-indexed in winter**

**Implication:**
- Deregulated hydro has **seasonally changing FX exposure**:
  - lower FX sensitivity in summer
  - higher FX sensitivity in winter due to linkage with thermal generation

---

## 3. Entity Pricing by Currency Logic

### 3.1 Directly USD-Linked or Contract-Based
- `renewable_ppa`
- `thermal_ppa`
- `import`
- `CfD_scheme`

These should be treated as strongly exchange-rate sensitive in GEL terms.

---

### 3.2 GEL-Denominated but Strongly FX-Sensitive
- `regulated_old_tpp`
- `regulated_new_tpp`

These are tariff-based categories in GEL, but their underlying economics are strongly USD-linked because of gas costs.

---

### 3.3 GEL-Denominated with Weak or Delayed FX Pass-Through
- `regulated_hpp`

These tariffs are fixed during the regulatory period and do not immediately reflect `xrate`.

---

### 3.4 GEL Market-Based Category
- `deregulated_hydro`

effect of xrate rapends if it is summer or winter period.

---

## 4. Mechanism of Exchange Rate Impact

### When GEL Depreciates (`xrate` increases)

#### GEL-Denominated Prices
GEL-denominated electricity prices usually increase because:
- USD-priced imports become more expensive in GEL
- USD-linked PPA and CfD electricity becomes more expensive in GEL
- regulated thermal tariffs increase because gas cost rises in GEL
- balancing electricity price can increase because USD-linked sources form a major share of balancing supply

#### USD-Denominated Prices
USD-denominated electricity prices usually change much less.

They may still move because:
- the generation composition changes
- GEL-priced components such as regulated hydro or deregulated hydro change their relative contribution

So, in USD terms, composition is usually the main driver, while pure exchange-rate noise is lower.

---


---

## 5. Analytical Rules

### For GEL Price Analysis
`xrate` is a **major factor** alongside:
- generation composition
- share of USD-linked supply
- thermal generation share
- balancing composition

### For USD Price Analysis
`xrate` has a **smaller direct effect**
because conversion noise is removed.

USD analysis is better for identifying:
- real cost changes
- composition effects
- structural market shifts

### For Comparing GEL vs USD Trends
- GEL prices show both composition effect and currency effect
- USD prices show composition effect more clearly
- divergence between GEL and USD price trends often indicates exchange-rate influence

---

## 6. Interaction with Main Price Categories

### Balancing Price
- strongly affected by USD-linked CfD/PPA, imports, and thermal costs
- highly sensitive to `xrate`
- especially important in the transitional market model because contract-based electricity is sold as balancing electricity
- See `balancing_price.md` for the canonical explanation of how balancing-market composition and supplier mix form the weighted-average balancing price

### Deregulated Price
- affected primarily by market structure and composition
- also indirectly affected by exchange rate because it competes with USD-linked sources

### Regulated Tariffs
- `regulated_old_tpp` and `regulated_new_tpp`:
  - GEL-denominated
  - but effectively USD-indexed through gas cost
- `regulated_hpp`:
  - GEL-denominated
  - weaker and delayed FX pass-through

---

## 7. Seasonality

Electricity prices and generation mix in Georgia have strong seasonal patterns (hydro vs thermal).

- Seasonality can amplify or mask exchange-rate effects
- For proper interpretation, compare:
  - yearly averages, or
  - same seasonal periods across years

See: **Seasonal Pattern** document for detailed analysis.

---

## 8. DB Structure

### `price_with_usd`
This view provides:
- `p_dereg_gel`
- `p_dereg_usd`
- `p_bal_gel`
- `p_bal_usd`
- `p_gcap_gel`
- `p_gcap_usd`
- `xrate`

Use this view for:
- direct exchange-rate analysis
- GEL vs USD comparison of deregulated, balancing, and guaranteed-capacity-related prices

### `tariff_with_usd`
This view provides:
- `tariff_gel`
- `tariff_usd`
- normalized `entity`
- `date`

Use this view for tariff analysis.

Important interpretation:
- `tariff_usd` is derived as `tariff_gel / xrate`
- for regulated thermal entities, this helps reveal that tariffs are GEL-denominated in form but economically USD-indexed / USD-driven
- for regulated hydro entities, `tariff_usd` is mainly an analytical conversion of a GEL-fixed tariff

---

## 10. Key Analytical Distinctions

- Do not treat all GEL-denominated prices as weakly exposed to currency
- Regulated thermal tariffs are GEL-denominated but strongly FX-sensitive
- CfD and PPA electricity is contract-based and strongly FX-sensitive
- In the current transitional model, CfD and PPA electricity is a core balancing-price driver
- USD-based analysis is useful for removing part of the exchange-rate noise and isolating structural price movements

