# Tariff Structure

## Definitions
- **PSO (Public Service Obligation):** Activities or entities subject to tariff regulation by GNERC.
- **Regulatory period:** Period for which GNERC sets tariffs.
- **Tariff:** Regulated electricity price, usually expressed in GEL/MWh unless otherwise stated.
- **Guaranteed Capacity Fee:** Fixed payment for thermal power plants for being ready to provide capacity to the system; expressed in GEL/day.

---

## Methodology
- Tariffs are approved by GNERC using a cost-plus methodology.
- GNERC approves tariffs for entities with public service obligation (PSO).
- Electricity transmission and distribution, as natural monopolies, are automatically considered PSO activities and GNERC approves their tariffs.
- Some electricity generation units are also PSO entities and are tariff regulated.
- The electricity market concept design defines a plan for deregulation of some hydro power plants. After deregulation, they no longer act as PSO entities and no tariff is set by the regulator.
- For tariff-regulated hydro power plants, the regulatory period is 3 years; i.e. the approved tariff is fixed for 3 years.
- For tariff-regulated thermal power plants, the guaranteed capacity fee is calculated on a yearly basis and expressed in GEL/day.
- For generated electricity from regulated thermal plants, no tariff is set fully in advance. After each month, the regulated variable price is defined based on actual cost and generated electricity, i.e. tariff = cost / generation.
- Because regulated thermal variable cost is recalculated based on actual monthly cost, exchange rate and gas price changes are reflected in tariff immediately.

---

## Components

### Hydro Tariffs
- Hydro tariffs mainly consist of fixed O&M and depreciation, with minimal variable costs.
- For hydro power plants, tariffs are set in advance and are GEL-denominated.
- If hydro power plants have USD-linked costs, they bear exchange rate risk within the regulatory period.
- Exchange rate effects for hydro are reflected only in the next regulatory period, not immediately.

**Simplified formula:**
- `tariff_gel = expected_average_total_cost / expected_generation`

---

### Thermal Tariffs
- **Fixed Component:** Guaranteed Capacity Fee. Covers fixed costs and is paid for every day the thermal power plant is ready to provide capacity to the system.
- The capacity fee is **not included** in the thermal per-MWh tariff.
- The guaranteed capacity fee is paid by all end-consumers proportionally to their consumption.
- **Variable Component:** Per-MWh fee depends mainly on natural gas cost and plant efficiency.
- This variable tariff is not fully approved in advance. Once the month is over and variable costs are known, the regulated price is calculated as total variable cost divided by generated MWh.
- In practice, variable cost is mainly natural gas cost, including transportation cost.
- Even if the final gas price is paid in GEL, it is strongly linked to USD-priced gas and therefore reflects exchange rate movements.

**Simplified formula:**
- `thermal_variable_tariff_gel = total_variable_cost_gel / generation_mwh`

**FX interpretation:**
- Gas price is USD-linked, so regulated thermal tariffs in GEL are highly correlated with exchange rate.
- Regulated thermal plants are therefore highly **FX-sensitive**, but they do not bear the same exchange rate risk as fixed-tariff hydro plants because the cost is passed through into the monthly tariff calculation.

---

## Tariff Entities

### Hydro Plants

#### State-owned tariff-regulated hydro power plants

| Entity | Label | Installed capacity | Approximate share in total installed capacity | Deregulation |
|---|---|---:|---:|---|
| `ltd "engurhesi"1` | Enguri HPP | 1300 MW | 26% | deregulation not anticipated |
| `ltd "vardnili hpp cascade"` | Vardnili HPP | 220 MW | 4.5% | deregulation not anticipated |

---

#### Energo-Pro owned tariff-regulated hydro power plants

| Entity | Label | Installed capacity | Approximate share in total installed capacity | Deregulation |
|---|---|---:|---:|---|
| `jsc "energo-pro georgia genration" (dzevrulhesi)` | Dzevruli HPP | 80 MW | 1.6% | expected from May 2026 |
| `jsc "energo-pro georgia genration" (gumathesi)` | Gumati HPP | 71.2 MW | 1.4% | deregulated from May 2024 |
| `jsc "energo-pro georgia genration" (shaorhesi)` | Shaori HPP | 40.4 MW | <1% | deregulated from Jan 2021 |
| `jsc "energo-pro georgia genration" (rionhesi)` | Rioni HPP | 54 MW | 1% | deregulated from May 2022 |
| `jsc "energo-pro georgia genration" (lajanurhesi)` | Lajanuri HPP | 115.6 MW | 2.3% | expected from Jan 2027 |

---

#### Other hydro plants with tariff regulation

| Entity | Label | Installed capacity | Approximate share in total installed capacity | Deregulation |
|---|---|---:|---:|---|
| `jsc "georgian water & power" (zhinvalhesi)` | Zhinvali HPP | 130 MW | 2.6% | deregulation not anticipated |
| `ltd "vartsikhe-2005"` | Vartsikhe HPP | 184 MW | 3.7% | deregulation not anticipated |
| `ltd "khrami_1"` | Khrami I HPP | 113.5 MW | 2.2% | expected from Jan 2027 |
| `ltd "khrami_2"` | Khrami II HPP | 110 MW | 2.2% | expected from Jan 2027 |

---

### Thermal Plants

#### State-owned tariff-regulated new thermal plants

| Entity | Label | Installed capacity | Approximate share in total installed capacity | Deregulation |
|---|---|---:|---:|---|
| `ltd "gardabni thermal power plant"` | Gardabani TPP | 231.2 MW | 4.7% | deregulation not anticipated |

---

#### Energo-Pro owned tariff-regulated thermal plants

| Entity | Label | Installed capacity | Approximate share in total installed capacity | Deregulation |
|---|---|---:|---:|---|
| `ltd "g power" (capital turbines)` | G-POWER | 110 MW | 2.2% | deregulation not anticipated |

---

#### Other tariff-regulated thermal plants

| Entity | Label | Installed capacity | Approximate share in total installed capacity | Deregulation |
|---|---|---:|---:|---|
| `ltd "mtkvari energy"` | Mtkvari Energy | 300 MW | 6% | deregulation not anticipated |
| `ltd "iec" (tbilresi)` | Tbilisi TPP | 272 MW | 5.5% | deregulation not anticipated |

---

## Notes
- All hydro power plants under tariff regulation were built before 1985 and their initial investment cost is already recovered.
- The regulated tariff is effectively an expected average total cost (ATC).
- Engurhesi is Georgia's largest hydro power plant.
- Thermal tariffs in GEL depend strongly on natural gas prices and generation regime (how efficiently the plant was operating).
- Mtkvari Energy, Tbilisi TPP (IEC), and G-POWER are regulated old TPPs with lower efficiency and higher marginal cost.
- Gardabani TPP is a regulated new TPP.
- Entity labels are provided for clearer chart legends and report outputs.

---

## Key Events
- **Jan 2024:** Natural gas procurement price for regulated thermal plants increased by up to 50% after being fixed at a low level for several years. This directly affected the regulated price of all regulated thermal plants.  
  **Note:** the gas price increase was only for regulated thermal plants. For thermal PPAs, it probably was not changed. This created direct upward pressure on balancing price via the thermal component.
- **May 2025:** Enguri HPP and Vardnili HPP Cascade tariffs increased due to a legislative amendment requiring these plants to cover the cost of electricity supplied to the occupied territory of Abkhazia.

---

## Tariff Dependencies
- **Enguri:** Reference hydro tariff — low and relatively stable.
- **Gardabani TPP:** New CCGT; tariff follows gas cost and exchange rate.
- **Old TPPs:** Less efficient; tariff follows gas cost and exchange rate more heavily.
- **Hydro plants:** Tariffs are stable during the regulatory period, but can contain embedded FX exposure if costs are USD-linked.
- **Thermal plants:** Tariffs react immediately to gas price and exchange rate changes through monthly recalculation.

---

## Analytical Implications
- Thermal tariffs rise with GEL depreciation and gas price increases.
- Hydro tariffs are stable across seasons and across the regulatory period.
- Guaranteed capacity ensures cost recovery even at low generation.
- Exchange rate immediately affects regulated thermal tariffs.
- For hydro, exchange rate effect is reflected only in the next regulatory period.
- When analyzing tariff behavior, hydro and thermal plants should be treated differently because hydro uses ex-ante fixed tariffs while thermal uses ex-post cost-based variable tariffs.

---

## Example Indicators
- Compare `tariff_gel` vs `xrate` to evaluate FX sensitivity of regulated thermal plants.
- Compare `tariff_usd` stability across regulated hydro plants to identify whether GEL-denominated tariff movements mainly reflect exchange rate changes.
- Track changes in thermal `tariff_gel` after gas price shocks.
- Compare old TPPs vs Gardabani TPP to assess efficiency-related tariff differences.

---

## Data Mapping

### Regulated Tariffs (Hydro & Thermal)
- View: `tariff_with_usd`
- Columns:
  - `tariff_gel` — regulated tariff in GEL
  - `tariff_usd` — GEL tariff converted using same-date exchange rate
- Entity: normalized via `entities.entity_normalized`
- Frequency: time-series (by date)

**Notes:**
- `tariff_usd` is a derived analytical value (not regulator-approved USD tariff)
- Thermal tariffs reflect monthly cost-based recalculation
- Hydro tariffs remain fixed within regulatory period