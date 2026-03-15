# Answer Templates

Structure templates by query type. Select the template matching the query's intent.

## Template: factual_lookup / single_value

**When**: User asks for a specific value ("What was balancing price in June 2024?")
**Length**: 50-150 words (1-2 sentences)
**Structure**:
- Direct answer with value + unit + brief context
- Example: "The balancing price in June 2024 was 142.3 GEL/MWh, which is lower than the winter average due to higher hydro generation share in summer months."

## Template: data_retrieval / list

**When**: User asks to show or list data ("Show monthly tariffs for Enguri in 2023")
**Length**: 100-300 words
**Structure**:
- Summary sentence with key finding
- 2-3 key observations referencing the data table
- Example: "Enguri tariff remained stable at 30.2 GEL/MWh throughout 2023. Key observations: [1] No tariff changes during the year, [2] This is consistent with GNERC's regulated cost-plus methodology."

## Template: data_explanation / driver analysis

**When**: User asks why something changed ("Why did balancing price increase in Nov 2021?")
**Length**: 300-800 words
**Structure**:

```
**[Question topic]: ანალიტიკური შეჯამება**

[Opening paragraph: state overall finding with numbers]

1. **[First Factor / გენერაციის სტრუქტურა]:**
   - [List 2-3 main share changes with EXACT numbers from data]
   - [Cite correlation if available: e.g., "კორელაცია -0.66"]
   - [Explain mechanism using domain knowledge]

2. **[Second Factor / გაცვლითი კურსი]:**
   - [Cite actual xrate change from data]
   - [Cite correlation if available: e.g., "კორელაცია 0.61"]
   - [Explain mechanism]

3. **[Seasonal Pattern]** (if applicable):
   - [Compare summer vs winter with specific numbers]
```

NO LENGTH RESTRICTION for analytical queries — provide comprehensive insights.

## Template: comparison

**When**: User asks to compare ("Compare summer and winter balancing prices")
**Length**: 150-400 words
**Structure**:
- Comparison frame (what is being compared, over what period)
- Side-by-side observations with specific numbers
- Key differentiators explaining the gap
- If seasonal: cite composition differences (hydro share in summer vs thermal/import in winter)

## Template: forecast

**When**: User asks for prediction/trend projection ("Forecast balancing price to 2032")
**Length**: 200-500 words
**Structure**:
- Forecast value with R² reliability indicator
- Methodology note ("Based on linear regression...")
- MANDATORY caveats (tiered by R² — see forecast-caveats.md)
- Non-extrapolatable factors
- Summer/winter separation if applicable

## Template: conceptual_definition

**When**: User asks about a concept ("What is a PPA?")
**Length**: 100-300 words
**Structure**:
- General definition (2-3 sentences)
- Georgia-specific context (2-3 sentences)
- Use domain knowledge; if topic is not covered, acknowledge the limitation and suggest external sources

## Missing domain knowledge response

When domain knowledge does not cover the user's question:
1. Acknowledge: "This specific information is not currently available in my domain knowledge base"
2. Suggest: "For current information about [topic], I recommend [specific source]"
3. Provide what you CAN say from available data
