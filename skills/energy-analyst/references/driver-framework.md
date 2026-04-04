# Driver Framework - Behavioral Rules

For detailed driver mechanics (xrate mechanism, entity pricing, composition effects), see domain knowledge topics: `balancing_price`, `currency_influence`.

## Driver Priority (MUST follow this order)

When analyzing balancing electricity prices, present drivers in this order:

1. **Composition (Entity Shares)** - primary driver for both GEL and USD prices
   - Must cite actual share values from data
   - Must cite correlation coefficient if available in statistics
   - For each major share shift, state whether that component was cheap or expensive relative to balancing price in that period
2. **Source Price / Tariff Layer** - required when source-price evidence is present
   - Cite exact values from `price_*`, `contribution_*`, or tariff columns
   - Regulated tariffs are quantity-weighted by balancing sales from `mv_balancing_trade_with_tariff`
   - Cover deregulated hydro plus regulated HPP/new TPP/old TPP layers when present
   - Compare each cited source price to balancing price in the same period
   - If source price is below balancing price, higher share means downward pressure; if above balancing price, higher share means upward pressure
   - If share falls, explain the reversed effect: loss of downward pressure or loss of upward pressure
   - Pair major share changes with the corresponding source-price change, not as separate disconnected bullets
   - Describe contribution columns as estimated decomposition, not exact causality
3. **Exchange Rate (XRate GEL/USD)** - critical for GEL price, smaller impact on USD price
   - Cite actual xrate change from data
   - Cite correlation if available
4. **Seasonal Patterns** (if applicable)
   - For long-term analysis: separate summer vs winter trends

## Analysis Structure

For price driver queries, structure the answer as:

1. **Composition**: List 2-3 main share changes with exact numbers. Cite correlation if available.
2. **Source Price / Tariff Layer**: Cite exact regulated tariffs and deregulated hydro prices when present. Compare them to balancing price in the same period. Cite `contribution_*` values when present. Explain whether each component was pushing the weighted average up or down. Explain the residual import/PPA/CfD layer carefully. CfD_scheme has a confidential price like PPAs — its influence is visible through `share_cfd_scheme` and the `share_ppa_import_total` residual.
3. **Exchange Rate**: Cite actual xrate change. Explain impact mechanism. Cite correlation if available.
4. **Seasonal Patterns** (if applicable): Compare summer vs winter composition and price levels.

## Causality Rules

- Use `observed`, `associated with`, `consistent with`, and `likely pressure` for observational data.
- Use stronger causal wording only when both a correlation coefficient and a documented mechanism are available.
- Source-price and contribution layers are analytical evidence, not a perfect decomposition of balancing price.
- Use numeric values only from `data preview`, `CAUSAL CONTEXT`, `COMPONENT PRESSURE SUMMARY`, `DERIVED ANALYSIS EVIDENCE`, or explicit tariff/source-price columns.
- Do not invent blended source averages, hidden import/PPA prices, or implied component prices unless those exact numbers are present in the evidence.
- For `deregulated_hydro`, always check season before generalizing: summer often lowers price, winter must be judged from actual `p_dereg_*` versus balancing price and the documented thermal-linkage mechanism.
- For regulated thermal layers, mention documented gas-price / xrate linkage when tariffs rise materially and those layers carry meaningful share.
