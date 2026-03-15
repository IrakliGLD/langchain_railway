# Driver Framework — Behavioral Rules

For detailed driver mechanics (xrate mechanism, entity pricing, composition effects), see domain knowledge topics: `balancing_price`, `currency_influence`.

## Driver Priority (MUST follow this order)

When analyzing balancing electricity prices, present drivers in this order:

1. **Composition (Entity Shares)** — PRIMARY DRIVER for BOTH GEL and USD prices
   - Must cite ACTUAL share values from data (e.g., "regulated HPP share increased from 22.3% to 35.7%")
   - Must cite correlation coefficient if available in statistics
2. **Exchange Rate (XRate GEL/USD)** — CRITICAL for GEL price, SMALL impact on USD price
   - Cite actual xrate change from data
   - Cite correlation if available
3. **Seasonal Patterns** (if applicable)
   - For long-term analysis: MUST separate summer vs winter trends

## Analysis Structure

For price driver queries, structure the answer as:

1. **Composition (გენერაციის სტრუქტურა)**: List 2-3 main share changes with exact numbers. Cite correlation if available.
2. **Exchange Rate (გაცვლითი კურსი)**: Cite actual xrate change. Explain impact mechanism. Cite correlation if available.
3. **Seasonal Patterns** (if applicable): Compare summer vs winter composition and price levels.

## Causality Rules

- Use "associated with", "consistent with", "likely pressure" for observational data
- Use causal language only when correlation coefficient is available AND mechanism is documented
- Correlation ≥ |0.5| with documented mechanism → acceptable causal language
- Correlation < |0.5| or no mechanism → observational language only
