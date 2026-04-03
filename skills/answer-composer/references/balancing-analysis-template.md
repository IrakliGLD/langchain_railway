# Balancing Analysis Template

The detailed format for balancing electricity price analysis queries. Use this for balancing price comparison, driver analysis, and explanation queries.

For entity definitions, pricing mechanisms, cost tiers, and xrate mechanism, see domain knowledge topics: `balancing_price`, `currency_influence`.

## Step-by-step data citation process

1. Look at the data preview and identify the exact rows for the periods being compared.
2. Extract exact values for both composition and price-driver columns.
3. Cite numbers directly. Do not describe a factor without numbers if the data contains numbers.

## Correlation citation

If `stats_hint` contains correlation coefficients, you must cite them.

- Cite the coefficient together with the driver it refers to.
- Do not use speculative wording when a correlation coefficient is available.
- Keep correlation language observational unless the mechanism is explicitly documented.

## Driver priority

Present drivers in this order (see `energy-analyst/references/driver-framework.md` for full rules):

1. **Composition** (shares of 7 entity categories) - primary driver
2. **Source Price / Tariff Layer** - required when `price_*`, `contribution_*`, or tariff columns are present
3. **Exchange Rate** (`xrate`) - critical for GEL, smaller for USD
4. **Seasonal patterns** - separate summer/winter for long-term analysis

## Source price / tariff layer

When the data contains source-price evidence, you must cite it explicitly instead of giving a composition-only explanation.

- Use `price_deregulated_hydro_gel` / `price_deregulated_hydro_usd` for deregulated hydro reference prices.
- Use `price_regulated_hpp_gel` / `price_regulated_hpp_usd` for regulated HPP reference prices.
- Use `price_regulated_new_tpp_gel` / `price_regulated_new_tpp_usd` for regulated new TPP reference prices.
- Use `price_regulated_old_tpp_gel` / `price_regulated_old_tpp_usd` for regulated old TPP reference prices.
- If `contribution_*` columns are present, cite them as estimated contribution layers to balancing price.
- If `residual_contribution_ppa_import_*` is present, describe it as the remaining unobserved import/PPA layer, not as a directly observed tariff.
- If source-price evidence is absent, skip this section instead of inventing tariffs or confidential PPA/import prices.
- Present this as observational decomposition, not exact causality.

## Comparison rules for balancing drivers

- For each major composition shift, compare both the share change and the component price change. Do not discuss share changes in isolation when `price_*`, `contribution_*`, or tariff columns are present.
- Compare each visible source price to the balancing price in the same period.
- If a component price is below balancing price, a higher share is downward pressure and a lower share removes downward pressure.
- If a component price is above balancing price, a higher share is upward pressure and a lower share removes upward pressure.
- Use `contribution_*` columns as the preferred summary of estimated pressure when they are present, then explain the direction with the share and price comparison.
- For `deregulated_hydro`, always check seasonality before generalizing:
  - summer: `p_dereg_*` is often low and usually pushes balancing price down
  - winter: compare actual `p_dereg_*` to balancing price and note the documented thermal-linkage mechanism when relevant
- For regulated thermal layers (`regulated_new_tpp`, `regulated_old_tpp`), if tariffs rise and the layer gains share, explicitly say this is upward pressure; mention documented gas-price / xrate linkage when relevant.
- For January 2024 or similar winter thermal increases, explain the mechanism, not just the direction: higher thermal tariffs raise the weighted average when thermal layers or thermally linked layers carry material share.

## Structured format

```text
**[Question topic]: analytical summary**

[Opening: state the overall balancing price change with exact numbers]

1. **Composition:**
   - [List 2-3 main share changes with exact values]
   - [Cover the main entity categories that changed]
   - [For each major share change, state whether the component is cheap or expensive relative to balancing price in that period]
   - [Cite correlation if available]
   - [For long-term analysis: compare summer vs winter composition]

2. **Source Price / Tariff Layer:**
   - [Cite exact source prices for deregulated hydro and regulated HPP/new TPP/old TPP when present]
   - [Compare each cited source price to the balancing price in the same period]
   - [State whether each component therefore pushed price up or down, or removed downward/upward pressure]
   - [Pair each major share change with the source-price change for that same component]
   - [Cite estimated `contribution_*` values when present]
   - [Explain which source layer became cheaper or more expensive]
   - [Describe `residual_contribution_ppa_import_*` as the remaining import/PPA layer]

3. **Exchange Rate:**
   - [Cite actual `xrate` change from the data]
   - [Explain the documented impact mechanism]
   - [Cite correlation if available]

4. **Seasonal Patterns (if applicable):**
   - [Separate summer vs winter dynamics]
```
