# Domain Focus Index

Maps query focus areas to the analytical rules, knowledge topics, and entity categories that apply.

## Focus: Balancing

**Trigger keywords**: balancing, p_bal, საბალანსო, баланс
**Knowledge topics**: balancing_price, market_structure
**Applicable rules**: entity-taxonomy (all 8 entities), driver-framework (full hierarchy), seasonal-rules (mandatory separation for >6 months)
**Key analysis**: composition shares + exchange rate + seasonal decomposition
**Entity pricing groups**: USD-priced (renewable_ppa, thermal_ppa, CfD_scheme, import) vs GEL-priced (deregulated_hydro, regulated_hpp, regulated_old_tpp, regulated_new_tpp)

## Focus: Tariff

**Trigger keywords**: tariff, ტარიფ, тариф, enguri, gardabani
**Knowledge topics**: tariffs
**Applicable rules**: confidentiality-rules (disclose regulated tariffs only), entity-taxonomy (regulated entities only)
**Key analysis**: GNERC cost-plus methodology, gas price linkage, exchange rate sensitivity
**Note**: Do NOT apply seasonal logic to tariff analyses. Focus on annual/multi-year trends.

## Focus: CPI / Inflation

**Trigger keywords**: cpi, inflation, ინფლაცია
**Knowledge topics**: general_definitions
**Applicable rules**: seasonal-rules (if comparing CPI to prices over time)
**Key analysis**: CPI category 'electricity_gas_and_other_fuels' trends. Frame as affordability comparison only if user asks.
**Note**: Only discuss electricity prices if user explicitly asks for affordability comparison.

## Focus: Generation

**Trigger keywords**: generation, technology, type_tech, demand, consumption, გენერ, потреб
**Knowledge topics**: generation_mix
**Applicable rules**: seasonal-rules (hydro vs thermal generation is seasonal), entity-taxonomy (generation types)
**Key analysis**: quantities (thousand MWh) by technology, shares, seasonal patterns
**Note**: Only discuss prices if user explicitly asks about price-generation relationships.

## Focus: Energy Security

**Trigger keywords**: energy security, უსაფრთხოება, independence, dependence
**Knowledge topics**: generation_mix, market_structure
**Applicable rules**: All
**Key analysis**:
- **Local/independent**: Hydro, Wind, Solar (no fuel imports)
- **Import-dependent**: Thermal (uses imported gas) + Direct electricity import
- Georgia's choice: import electricity OR import gas to generate electricity
- NEVER say "thermal reduces import dependence" — thermal relies on imported gas
- True energy security comes from renewables expansion
- Use `tech_quantity_view`: sum thermal + import as import-dependent; sum hydro + wind + solar as local; calculate local_share = local / (local + import_dependent)
