# Seasonal Rules — Behavioral Rules

For season definitions (months 4-7 summer, 8-12+1-3 winter), seasonal trends, and price comparison mechanics, see domain knowledge topic: `seasonal_patterns`.

## Mandatory Seasonal Separation

- Any analysis spanning more than 6 months MUST separate summer vs winter
- Monthly comparisons within a single season do not need separation
- Annual averages should note seasonal composition differences

## Seasonal-Adjusted Statistics Authority

When stats_hint contains `SEASONAL-ADJUSTED TREND ANALYSIS`, those statistics are AUTHORITATIVE:

1. Use the "Overall growth" percentage for multi-year trends
   - DO NOT compare first month to last month directly
   - DO NOT say "doubled" or "tripled" based on raw monthly data
   - USE the calculated CAGR (average annual growth rate)

2. Pay attention to incomplete year warnings
   - If stats say "Last year has only X months" → mention this caveat
   - DO NOT treat incomplete years as full years in trend analysis

3. For trend queries cite:
   - Year range: "From [first_year] to [last_year]"
   - Overall growth: "increased by [overall_growth_pct]%"
   - CAGR: "average annual growth of [cagr]%"
   - Seasonal pattern: "peak in [peak_month], low in [low_month]"

4. Distinguish between:
   - Long-term trend (use CAGR from stats)
   - Seasonal variation (use peak_month/low_month from stats)
   - Recent momentum (use recent_12m_growth if available)

## Seasonal Forecast Queries

For forecast/trend queries with seasonal split:
- Return MONTHLY data WITH a season column
- DO NOT aggregate by season (no GROUP BY season) — this loses time series data
- The Python visualization layer calculates separate trendlines for summer/winter
- ALWAYS separate summer and winter forecasts (different driver mixes)
