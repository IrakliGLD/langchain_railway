# Entity Taxonomy — Behavioral Rules

For full entity definitions, pricing mechanisms, and cost tiers, see domain knowledge topics: `balancing_price`, `currency_influence`.

## Rules

- Always analyze ALL 8 balancing entity categories — never omit any.
- The 8 observable categories are: `deregulated_ren`, `regulated_hpp`, `regulated_new_tpp`, `regulated_old_tpp`, `renewable_ppa`, `thermal_ppa`, `CfD_scheme`, `import`.
- `deregulated_ren` is the canonical identifier for the **deregulated renewable** category. It is the derived Trade residual after subtracting regulated generation, PPAs, import, and CfD quantities from the configured broad Trade total. Use `share_deregulated_ren` for its balancing-market share. Do not use the former hydro-only identifier or label.
- `CfD_scheme`: support-scheme generation (renewable-like), USD-priced, price is confidential. Structurally present in balancing (mandatory sale on balancing segment, same as PPAs). Treated as moderate-cost, USD-linked in share analysis. Its volume is included in `share_ppa_import_total` residual; its individual share is tracked as `share_cfd_scheme`.
- Distinguish cheap sources (regulated_hpp, deregulated_ren) from expensive sources (import, thermal_ppa) and USD-linked support schemes (renewable_ppa, CfD_scheme, thermal_ppa) when explaining price impacts.
- Support schemes = PPA + CfD ONLY. Regulated tariffs (HPP, old/new TPP) are NOT support schemes — they are cost-plus regulation. Guaranteed capacity for old thermals is a separate mechanism.

## Data Quality Warning

If shares show NULL or 0 for periods before 2020, this means data was NOT collected — NOT that the share was zero. NEVER say "share was 0%" for pre-2020 periods. Instead say "data is not available for this period."
