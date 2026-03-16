# Tariff Structure

## Methodology
Tariffs are approved by GNERC using cost-plus methodology.

## Components

### Hydro Tariffs
Mainly fixed O&M and depreciation; minimal variable costs.

### Thermal Tariffs
- **Fixed Component:** Guaranteed Capacity Fee (covers fixed costs)
- **Variable Component:** Per-MWh fee depends on gas price and efficiency
- **FX Exposure:** Gas price in USD → tariff_gel correlates with xrate

## Tariff Entities

### Hydro Plants

**Enguri HPP:**
- Entity: `ltd "engurhesi"1`
- Label: "Enguri HPP"

**Energo-Pro Hydro Plants:**

| Entity | Label |
|---|---|
| `jsc "energo-pro georgia genration" (dzevrulhesi)` | Dzevruli HPP |
| `jsc "energo-pro georgia genration" (gumathesi)` | Gumati HPP |
| `jsc "energo-pro georgia genration" (shaorhesi)` | Shaori HPP |
| `jsc "energo-pro georgia genration" (rionhesi)` | Rioni HPP |
| `jsc "energo-pro georgia genration" (lajanurhesi)` | Lajanuri HPP |

**Other Hydro:**

| Entity | Label |
|---|---|
| `jsc "georgian water & power" (zhinvalhesi)` | Zhinvali HPP |
| `ltd "vardnili hpp cascade"` | Vardnili HPP Cascade |
| `ltd "vartsikhe-2005"` | Vartsikhe HPP |
| `ltd "khrami_1"` | Khrami I HPP |
| `ltd "khrami_2"` | Khrami II HPP |

### Thermal Plants

| Entity | Label |
|---|---|
| `ltd "gardabni thermal power plant"` | Gardabani TPP |
| `ltd "mtkvari energy"` | Mtkvari Energy |
| `ltd "iec" (tbilresi)` | Tbilisi TPP |
| `ltd "g power" (capital turbines)` | G-POWER |

### Notes
- Engurhesi is Georgia's main large hydro plant; used as a reference for hydro-tariff correlation
- Thermal tariffs depend strongly on natural gas prices
- Mtkvari Energy, Tbilisi TPP (IEC), G-POWER are regulated old TPPs
- Gardabani TPP is a regulated new TPP
- Energo-Pro hydro plants (Rioni, Lajanuri, Shaori, Gumati, Dzevruli) have similar cost structures
- Entity labels are provided for clearer chart legends and report outputs

## Key Events

- **Jan 2024:** Natural gas procurement price for regulated thermal plants increased substantially after being fixed at a low level for several years. GNERC raised thermal tariffs (Gardabani, old TPPs) accordingly. Direct upward pressure on balancing price via thermal component.
- **May 2025:** Enguri HPP and Vardnili HPP Cascade tariffs increased — legislative amendment requires these plants to cover cost of electricity supplied to the occupied territory of Abkhazia.

## Tariff Context

- Tariff increases in 2024–2025 are primarily cost-driven, reflecting gas price rises, currency depreciation, and compensation mechanisms for unreimbursed energy
- Renewable PPAs generally have higher fixed tariffs than average summer balancing prices; as renewable share grows and cheap hydro share declines, summer balancing prices converge toward average PPA prices
- Balancing electricity is the residual of total generation minus volumes sold under bilateral contracts or on exchanges
- Seasonal price differences must be read in context of regulatory cost adjustments and evolving generation mix

## Tariff Dependencies
- **Enguri:** Reference hydro tariff — low, stable
- **Gardabani TPP:** New CCGT; tariff follows gas cost and xrate
- **Old TPPs:** Less efficient; higher tariffs, more volatile
- **Usage Hint:** Compare tariff_gel with p_bal_gel to assess regulatory lag

## Analytical Implications
- Thermal tariffs rise with GEL depreciation and gas price increases
- Hydro tariffs are stable across seasons
- Guaranteed capacity ensures cost recovery even at low generation

## Example Indicators
- Compare `tariff_gel` vs `xrate` to evaluate FX sensitivity
- Compare `p_gcap_gel` vs `tariff_gel` to separate fixed vs variable cost effect

## Data Sources
- **Regulated HPP tariff:** `tariff_with_usd` view (tariff_gel, tariff_usd columns)
- **Regulated thermal tariff:** `tariff_with_usd` view (Gardabani, old TPPs) — GEL tariffs reflecting xrate
- **Deregulated hydro price:** `price_with_usd` view (p_dereg_gel, p_dereg_usd)
- **Renewable PPA price:** USD-priced, NOT IN DATABASE — market-based
- **Thermal PPA price:** USD-priced, NOT IN DATABASE — market-based
- **Import price:** USD-priced, NOT IN DATABASE — market-based
