# Currency Influence

## GEL/USD Exchange Rate Effect
Electricity price analysis must consider GEL/USD exchange rate effects.

## USD-Denominated Costs
- Natural gas for thermal generation is priced in USD
- Electricity imports are priced in USD
- Regulated tariffs are adjusted for exchange-rate changes to cover USD-denominated costs

## Seasonality Hint
Compare yearly averages to neutralize strong seasonal swings (hydro vs thermal).

## Entity Pricing by Currency

### USD-priced entities:
- renewable_ppa
- thermal_ppa
- import

### GEL-priced entities:
- deregulated_hydro
- regulated_hpp
- regulated_old_tpp (GEL tariffs that directly reflect current xrate)
- regulated_new_tpp (GEL tariff that directly reflect current xrate)

## Mechanism
When GEL depreciates (xrate increases):
- **GEL price rises significantly:** All USD-priced entities convert at higher xrate, plus GEL-priced entities
- **USD price rises slightly:** Only GEL-priced entities like deregulated_hydro and regulated_hpp are affected
- The impact on USD price is SMALL because GEL-priced entity shares are very small
- Regulated old/new TPP tariffs adjust with xrate, affecting both GEL and USD prices

## Data Source
`price_with_usd` view, column: `xrate`

## Analysis Requirements
- For GEL price analysis: xrate is a MAJOR factor alongside composition
- For USD price analysis: xrate has SMALL impact (through GEL-priced entities), composition is PRIMARY driver
- When comparing GEL vs USD price trends: USD price shows composition effect with minimal xrate noise

## CPI Data
- **Data Source:** `monthly_cpi_mv`
- **Filter:** `cpi_type = 'electricity_gas_and_other_fuels'`
- **Source organization:** GEOSTAT (National Statistics Office of Georgia)
