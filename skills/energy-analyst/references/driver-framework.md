# Driver Framework - Behavioral Rules

For detailed driver mechanics (xrate mechanism, entity pricing, composition effects), see domain knowledge topics: `balancing_price`, `currency_influence`.

## Driver Priority (MUST follow this order)

When analyzing balancing electricity prices, present drivers in this order:

1. **Composition (Entity Shares)** - primary driver for both GEL and USD prices
   - Must cite actual share values from data
   - Must cite correlation coefficient if available in statistics
2. **Source Price / Tariff Layer** - required when source-price evidence is present
   - Cite exact values from `price_*`, `contribution_*`, or tariff columns
   - Cover deregulated hydro plus regulated HPP/new TPP/old TPP layers when present
   - Describe contribution columns as estimated decomposition, not exact causality
3. **Exchange Rate (XRate GEL/USD)** - critical for GEL price, smaller impact on USD price
   - Cite actual xrate change from data
   - Cite correlation if available
4. **Seasonal Patterns** (if applicable)
   - For long-term analysis: separate summer vs winter trends

## Analysis Structure

For price driver queries, structure the answer as:

1. **Composition**: List 2-3 main share changes with exact numbers. Cite correlation if available.
2. **Source Price / Tariff Layer**: Cite exact regulated tariffs and deregulated hydro prices when present. Cite `contribution_*` values when present. Explain the residual import/PPA layer carefully.
3. **Exchange Rate**: Cite actual xrate change. Explain impact mechanism. Cite correlation if available.
4. **Seasonal Patterns** (if applicable): Compare summer vs winter composition and price levels.

## Causality Rules

- Use `observed`, `associated with`, `consistent with`, and `likely pressure` for observational data.
- Use stronger causal wording only when both a correlation coefficient and a documented mechanism are available.
- Source-price and contribution layers are analytical evidence, not a perfect decomposition of balancing price.
