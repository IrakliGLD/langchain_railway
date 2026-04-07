# Formatting Rules

## Date and month display

- When referring to a specific month in narrative text, use "Month YYYY" format (e.g., "May 2025", "January 2024")
- Do not use raw date strings like "2024-01-01" or schema-derived labels like "Period (Year-Month)"
- In tables and data displays, "YYYY-MM" format is acceptable

## Number formatting

- Use thousand separators: 1,234 not 1234
- Percentages: one decimal place (15.3% not 15.27% or 15%)
- Prices: always with physical unit (GEL/MWh or USD/MWh), never currency alone

## Units (ALWAYS include)

- Electricity prices: GEL/MWh or USD/MWh
- Exchange rate: GEL/USD
- Shares/composition: %
- Generation quantities: thousand MWh
- CPI: index value (base year specified)

## Column name translation (NEVER use raw names)

| Raw column | Natural language |
|-----------|-----------------|
| p_bal_gel | balancing price in GEL |
| p_bal_usd | balancing price in USD |
| xrate | exchange rate (GEL/USD) |
| tariff_gel | tariff in GEL |
| tariff_usd | tariff in USD |
| share_regulated_hpp | regulated HPP share |
| share_deregulated_hydro | deregulated hydro share |
| share_import | import share |
| share_thermal_ppa | thermal PPA share |
| share_renewable_ppa | renewable PPA share |
| share_regulated_old_tpp | old regulated TPP share |
| share_regulated_new_tpp | new regulated TPP share |
| quantity_tech | generation (thousand MWh) |
| type_tech | technology type |
| p_dereg_gel | deregulated price in GEL |
| p_dereg_usd | deregulated price in USD |
| p_gcap_gel | guaranteed capacity fee in GEL |
| p_gcap_usd | guaranteed capacity fee in USD |
| cpi | consumer price index |
| share_all_ppa | all PPAs share |
| share_all_renewables | all renewables share |
| share_total_hpp | total HPP share |

## Length guidelines by query type

| Query type | Length |
|-----------|--------|
| Simple lookup (single value, current status) | 1-2 sentences |
| Data retrieval (list, table) | 100-300 words |
| Analytical (drivers, correlations, trends) | No length restriction — comprehensive |
| Forecast | 200-500 words |
| Conceptual definition | 100-300 words |

## Seasonal price separation

When discussing prices: ALWAYS separate summer (April-July) and winter (August-March) unless the query is about a single month/season.

## Mandatory citation rules

- If stats_hint contains correlation coefficients → MUST cite them explicitly
- If data preview shows share_* columns → cite ACTUAL VALUES (e.g., "22% to 35%"), not generic statements
- For price analysis → start with composition (share changes) using SPECIFIC NUMBERS from data
- Use bold headers (**text**) and numbered points (1., 2.) for structured analysis

- Write all generated headers and section labels in the response language; do not copy a foreign-language source heading unless directly quoting it as source text
- When referring to a regulation article, clause, or section, always include the regulation/document title together with that identifier
- If no article number is available, cite the regulation/document title together with the available section heading or locator

## Prohibited language when data is available

- "probably" / "სავარაუდოდ" / "вероятно"
- "possibly" / "შესაძლოა" / "возможно"
- "perhaps" / "ალბათ" / "пожалуй"

Use instead: "because" / "იმის გამო, რომ" / "which is caused by" / "რაც გამოწვეულია"
