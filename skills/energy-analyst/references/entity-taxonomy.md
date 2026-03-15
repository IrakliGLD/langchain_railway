# Entity Taxonomy — Behavioral Rules

For full entity definitions, pricing mechanisms, and cost tiers, see domain knowledge topics: `balancing_price`, `currency_influence`.

## Rules

- Always analyze ALL 7 balancing entity categories — never omit any.
- Distinguish cheap sources (regulated_hpp, deregulated_hydro) from expensive sources (import, thermal_ppa, renewable_ppa) when explaining price impacts.
- Support schemes = PPA + CfD ONLY. Regulated tariffs (HPP, old/new TPP) are NOT support schemes — they are cost-plus regulation. Guaranteed capacity for old thermals is a separate mechanism.

## Data Quality Warning

If shares show NULL or 0 for periods before 2020, this means data was NOT collected — NOT that the share was zero. NEVER say "share was 0%" for pre-2020 periods. Instead say "data is not available for this period."
