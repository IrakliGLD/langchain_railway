# Forecast Caveats

For detailed non-extrapolatable factors and forecasting challenges, see domain knowledge topic: `seasonal_patterns`.

## Trendline citation rule

If stats_hint contains "TRENDLINE FORECASTS", YOU MUST cite the forecast values explicitly:
- Use the forecast value from stats_hint, NOT guesses or calculations
- Include the R² value to indicate forecast reliability
- Format: "Based on linear regression (R²={r_squared}), the price is forecast to reach {forecast_value} GEL/MWh by {target_year}"
- NEVER say "forecast is the same as current" unless the trendline slope is actually near zero

## R²-based caveat templates (MANDATORY — include AFTER presenting the forecast)

### R² < 0.5 (low reliability)

"This forecast has moderate-to-low reliability (R²={r_squared}) due to variability in historical prices. Actual prices may differ significantly due to regulatory decisions (gas prices, tariffs), new PPA capacity, market rule changes, or import price shifts."

### R² ≥ 0.5 but < 0.7 (moderate reliability)

"This forecast assumes current market structure, PPA contracts, and regulatory framework remain unchanged. Actual prices may differ due to: gas price negotiations, new PPA/CfD capacity additions, GNERC tariff decisions, or changes in neighboring electricity markets."

### R² ≥ 0.7 (high reliability)

"While this trend is statistically strong (R²={r_squared}), it reflects past patterns and assumes unchanged regulatory and contractual conditions. Key uncertainties: (1) PPA/CfD capacity growth beyond current projections, (2) gas price negotiations with Azerbaijan, (3) potential market rule changes, (4) import price dynamics from neighboring markets."

## Forecasting best practices

| Horizon | Approach |
|---------|---------|
| Short-term (1-2 years) | Trendline + regulatory uncertainty caveat |
| Medium-term (3-5 years) | Trendline + scenario discussion (upside/downside from policy changes) |
| Long-term (5+ years) | Focus on structural drivers rather than linear extrapolation |

ALWAYS separate summer and winter forecasts (different driver mixes).
