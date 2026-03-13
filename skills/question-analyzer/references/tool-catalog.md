# Tool Catalog

Runtime source of truth: [contracts/question_analysis_catalogs.py](../../../contracts/question_analysis_catalogs.py)

## get_prices

- Concepts: balancing price, deregulated price, guaranteed capacity price, exchange rate, GEL, USD, trends
- Use for: price or exchange-rate retrieval over time or for a stated period
- Do not use for: conceptual definitions, tariff questions, share/composition-only questions
- Main params: `metric`, `currency`, `granularity`, `start_date`, `end_date`

## get_tariffs

- Concepts: regulated tariffs, GNERC-regulated entities, Enguri, Gardabani
- Use for: tariff lookups and tariff comparisons
- Do not use for: balancing price questions, conceptual definitions, generation mix
- Main params: `entities`, `currency`, `start_date`, `end_date`

## get_generation_mix

- Concepts: generation, technology mix, hydro, thermal, wind, solar, quantity, shares
- Use for: generation mix or quantity by type
- Do not use for: tariffs, balancing price, conceptual definitions
- Main params: `types`, `mode`, `granularity`, `start_date`, `end_date`

## get_balancing_composition

- Concepts: balancing shares, composition, imports, PPAs, hydro shares
- Use for: balancing market composition and share questions
- Do not use for: price-only questions, conceptual definitions, tariffs
- Main params: `entities`, `start_date`, `end_date`
