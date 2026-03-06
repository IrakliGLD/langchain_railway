# Generation Mix and Energy Security

## Generation by Technology

### Data Source
Table: `tech_quantity_view`
- Columns: `type_tech`, `quantity` (thousand MWh), `time_month`
- Technologies: hydro, thermal, wind, solar, import, export, supply-distribution, direct customers, losses, abkhazeti

### Total Demand Calculation
Total electricity demand = SUM of: abkhazeti + supply-distribution + direct customers + losses + export

### Total Generation Calculation
Total domestic generation = SUM of: hydro + thermal + wind + solar

## Energy Security Analysis

**CRITICAL FACT:** Thermal generation uses imported natural gas and cannot be considered fully domestic/local generation.

### Correct Classification

**Local Generation (NO import dependence):**
- Hydro (all types: regulated HPP, deregulated hydro, reservoir, run-of-river)
- Wind (renewable, no fuel imports)
- Solar (renewable, no fuel imports)

**Import-Dependent Generation:**
- Thermal (uses imported natural gas for power generation)
- Direct electricity import
- Note: Both depend on cross-border energy supply

### Analytical Implications
- When analyzing energy security, thermal is NOT a substitute for imports — it IS import-dependent
- The real choice for Georgia is: import electricity OR import gas to generate electricity
- True energy independence comes from hydro, wind, and solar expansion
- Winter import dependence = direct electricity imports + gas imports for thermal generation
- Summer energy surplus is real because it's based on local hydro without fuel imports

### Example Statements
- ✅ CORRECT: "Georgia's energy security depends on local renewables (hydro, wind, solar). Thermal generation, while domestic, relies on imported gas and does not reduce import dependence."
- ✅ CORRECT: "In winter, Georgia is import-dependent: ~30% direct electricity import + thermal generation using imported gas."
- ❌ WRONG: "Thermal generation is local production that reduces import dependence."
- ❌ WRONG: "Georgia can achieve energy independence by increasing thermal capacity."

## Energy Balance
**Data Source:** `energy_balance_long_mv` (from GEOSTAT)
- Use yearly aggregation (not monthly)
- Contains national energy balances and sectoral demand indicators
