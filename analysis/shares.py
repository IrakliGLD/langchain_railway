"""
Share calculations and price decomposition for energy market analysis.

Handles:
- Entity share calculations (import, hydro, thermal, PPAs)
- Balancing price correlation analysis with driver variables
- Entity-level price contribution decomposition
- Weighted balancing price contributions by month
"""
import logging
from typing import Any, Optional

import pandas as pd
from sqlalchemy import text

from config import MAX_ROWS
from core.query_executor import check_dataframe_memory

log = logging.getLogger("Enai")


def build_balancing_correlation_df(conn: Any) -> pd.DataFrame:
    """
    Returns a monthly panel with balancing price and driver variables.

    Variables included:
    - Targets: p_bal_gel, p_bal_usd
    - Drivers: xrate, entity shares (import, deregulated_hydro, regulated_hpp,
               renewable_ppa, thermal_ppa), tariffs (Enguri, Gardabani, old TPPs)

    CRITICAL: Shares are calculated using ONLY the canonical balancing
    segment filter LOWER(REPLACE(segment, ' ', '_')) = 'balancing' to
    properly reflect the composition affecting balancing price.

    Args:
        conn: SQLAlchemy connection object

    Returns:
        DataFrame with monthly panel of prices, shares, and tariffs

    Examples:
        >>> from sqlalchemy import create_engine
        >>> engine = create_engine(DB_URL)
        >>> with engine.connect() as conn:
        ...     df = build_balancing_correlation_df(conn)
        ...     print(df[['date', 'p_bal_gel', 'share_import']].head())
    """
    sql = """
    WITH shares AS (
      SELECT
        t.date,
        SUM(t.quantity) AS total_qty,
        SUM(CASE WHEN t.entity = 'import' THEN t.quantity ELSE 0 END) AS qty_import,
        SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity ELSE 0 END) AS qty_dereg_hydro,
        SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity ELSE 0 END) AS qty_reg_hpp,
        SUM(CASE WHEN t.entity = 'regulated_new_tpp' THEN t.quantity ELSE 0 END) AS qty_reg_new_tpp,
        SUM(CASE WHEN t.entity = 'regulated_old_tpp' THEN t.quantity ELSE 0 END) AS qty_reg_old_tpp,
        SUM(CASE WHEN t.entity = 'renewable_ppa' THEN t.quantity ELSE 0 END) AS qty_ren_ppa,
        SUM(CASE WHEN t.entity = 'thermal_ppa' THEN t.quantity ELSE 0 END) AS qty_thermal_ppa
      FROM trade_derived_entities t
      WHERE LOWER(REPLACE(t.segment, ' ', '_')) = 'balancing'
      GROUP BY t.date
    ),
    tariffs AS (
      SELECT
        date,
        MAX(CASE
          WHEN entity = 'ltd "engurhesi"1'
          THEN tariff_gel
        END) AS enguri_tariff_gel,
        MAX(CASE
          WHEN entity = 'ltd "gardabni thermal power plant"'
          THEN tariff_gel
        END) AS gardabani_tpp_tariff_gel,
        AVG(CASE
          WHEN entity IN (
            'ltd "mtkvari energy"',
            'ltd "iec" (tbilresi)',
            'ltd "g power" (capital turbines)'
          )
          THEN tariff_gel
        END) AS grouped_old_tpp_tariff_gel
      FROM tariff_with_usd
      GROUP BY date
    )
    SELECT
      p.date,
      p.p_bal_gel,
      p.p_bal_usd,
      p.xrate,
      (s.qty_import / NULLIF(s.total_qty,0)) AS share_import,
      (s.qty_dereg_hydro / NULLIF(s.total_qty,0)) AS share_deregulated_hydro,
      (s.qty_reg_hpp / NULLIF(s.total_qty,0)) AS share_regulated_hpp,
      (s.qty_reg_new_tpp / NULLIF(s.total_qty,0)) AS share_regulated_new_tpp,
      (s.qty_reg_old_tpp / NULLIF(s.total_qty,0)) AS share_regulated_old_tpp,
      (s.qty_ren_ppa / NULLIF(s.total_qty,0)) AS share_renewable_ppa,
      (s.qty_thermal_ppa / NULLIF(s.total_qty,0)) AS share_thermal_ppa,
      ((s.qty_ren_ppa + s.qty_thermal_ppa) / NULLIF(s.total_qty,0)) AS share_all_ppa,
      ((s.qty_dereg_hydro + s.qty_reg_hpp + s.qty_ren_ppa) / NULLIF(s.total_qty,0)) AS share_all_renewables,
      tr.enguri_tariff_gel,
      tr.gardabani_tpp_tariff_gel,
      tr.grouped_old_tpp_tariff_gel
    FROM price_with_usd p
    LEFT JOIN shares s ON s.date = p.date
    LEFT JOIN tariffs tr ON tr.date = p.date
    ORDER BY p.date
    """
    # Add LIMIT using configured MAX_ROWS
    sql_with_limit = f"{sql.strip()}\nLIMIT {MAX_ROWS};"
    res = conn.execute(text(sql_with_limit))
    df = pd.DataFrame(res.fetchall(), columns=list(res.keys()))
    check_dataframe_memory(df)
    return df


def compute_weighted_balancing_price(conn: Any) -> pd.DataFrame:
    """
    Compute each month's contribution to the grand-total weighted-average balancing price.

    Returns monthly panel with:
    - date: Month
    - p_bal_gel, p_bal_usd: Monthly balancing price (actual weighted average)
    - contribution_gel, contribution_usd: Month's contribution to all-time average

    Formula: (monthly_price * monthly_quantity) / total_quantity_across_all_months

    Note: The monthly weighted average price is already in price_with_usd.
    This function calculates how much each month contributes to the grand average.

    CRITICAL: Uses ONLY balancing segment to calculate weights.
    Uses case-insensitive segment matching.

    Args:
        conn: SQLAlchemy connection object

    Returns:
        DataFrame with monthly contributions to weighted average

    Examples:
        >>> with engine.connect() as conn:
        ...     df = compute_weighted_balancing_price(conn)
        ...     total_contribution = df['contribution_gel'].sum()
        ...     print(f"Total should equal weighted average: {total_contribution:.2f}")
    """
    sql = """
    WITH t AS (
      SELECT date, entity, SUM(quantity) AS qty
      FROM trade_derived_entities
      WHERE LOWER(REPLACE(segment, ' ', '_')) = 'balancing'
        AND entity IN ('deregulated_hydro','import','regulated_hpp',
                       'regulated_new_tpp','regulated_old_tpp',
                       'renewable_ppa','thermal_ppa')
      GROUP BY date, entity
    ),
    w AS (SELECT date, SUM(qty) AS total_qty FROM t GROUP BY date)
    SELECT
      p.date,
      p.p_bal_gel,
      p.p_bal_usd,
      (p.p_bal_gel * w.total_qty) / NULLIF(SUM(w.total_qty) OVER (),0) AS contribution_gel,
      (p.p_bal_usd * w.total_qty) / NULLIF(SUM(w.total_qty) OVER (),0) AS contribution_usd
    FROM price_with_usd p
    JOIN w ON w.date = p.date
    ORDER BY p.date;
    """
    res = conn.execute(text(sql))
    df = pd.DataFrame(res.fetchall(), columns=list(res.keys()))
    check_dataframe_memory(df)
    return df


def compute_entity_price_contributions(
    conn: Any,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Decompose balancing price into entity-level contributions.

    Returns monthly panel showing:
    - Balancing price (actual weighted average)
    - Each entity's share in balancing electricity
    - Entity-level reference prices (where available)
    - Estimated contribution to balancing price: share × reference_price

    CRITICAL NOTES:
    - Regulated entities (regulated_hpp, regulated TPPs): use tariff_gel from tariff_with_usd
    - Deregulated hydro: use p_dereg_gel from price_with_usd
    - PPAs and imports: reference prices NOT available in database (confidential), but reference number are provided in domain_knowldege.py
    - Actual balancing transaction prices differ from reference prices
    - This provides directional insight, not exact decomposition

    Use this to explain:
    - Which entities drove price changes month-over-month
    - How composition shifts affected weighted average price
    - Relative contribution of cheap vs expensive sources

    Args:
        conn: SQLAlchemy connection object

    Returns:
        DataFrame with entity shares, reference prices, and estimated contributions

    Examples:
        >>> with engine.connect() as conn:
        ...     df = compute_entity_price_contributions(conn)
        ...     print(df[['date', 'balancing_price_gel', 'share_import',
        ...              'contribution_deregulated_hydro_gel']].head())
    """
    share_filters = []
    price_filters = []
    params = {}
    if start_date:
        share_filters.append("t.date >= :start_date")
        price_filters.append("p.date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        share_filters.append("t.date <= :end_date")
        price_filters.append("p.date <= :end_date")
        params["end_date"] = end_date

    share_where = " AND ".join(
        ["LOWER(REPLACE(t.segment, ' ', '_')) = 'balancing'"] + share_filters,
    )
    price_where = (
        "WHERE " + " AND ".join(price_filters)
        if price_filters
        else "WHERE p.date >= '2015-01-01'"
    )
    if price_filters:
        price_where += " AND p.date >= '2015-01-01'"

    sql = f"""
    WITH shares AS (
      SELECT
        t.date,
        SUM(t.quantity) AS total_qty,
        SUM(CASE WHEN t.entity = 'import' THEN t.quantity ELSE 0 END) AS qty_import,
        SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity ELSE 0 END) AS qty_dereg_hydro,
        SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity ELSE 0 END) AS qty_reg_hpp,
        SUM(CASE WHEN t.entity = 'regulated_new_tpp' THEN t.quantity ELSE 0 END) AS qty_reg_new_tpp,
        SUM(CASE WHEN t.entity = 'regulated_old_tpp' THEN t.quantity ELSE 0 END) AS qty_reg_old_tpp,
        SUM(CASE WHEN t.entity = 'renewable_ppa' THEN t.quantity ELSE 0 END) AS qty_ren_ppa,
        SUM(CASE WHEN t.entity = 'thermal_ppa' THEN t.quantity ELSE 0 END) AS qty_thermal_ppa
      FROM trade_derived_entities t
      WHERE {share_where}
      GROUP BY t.date
    ),
    entity_prices AS (
      SELECT
        p.date,
        -- Reference prices from available sources
        p.p_dereg_gel AS price_deregulated_hydro_gel,
        p.p_dereg_usd AS price_deregulated_hydro_usd,

        -- Regulated HPP: use weighted average of main HPPs or Enguri as proxy
        -- Using ILIKE for case-insensitive matching (PostgreSQL extension)
        (SELECT AVG(t1.tariff_gel)
         FROM tariff_with_usd t1
         WHERE t1.date = p.date
           AND (t1.entity ILIKE '%engurhesi%'
                OR t1.entity ILIKE '%energo-pro%'
                OR t1.entity ILIKE '%vardnili%')
        ) AS price_regulated_hpp_gel,
        (SELECT AVG(t1.tariff_usd)
         FROM tariff_with_usd t1
         WHERE t1.date = p.date
           AND (t1.entity ILIKE '%engurhesi%'
                OR t1.entity ILIKE '%energo-pro%'
                OR t1.entity ILIKE '%vardnili%')
        ) AS price_regulated_hpp_usd,

        -- Regulated new TPP: Gardabani
        (SELECT t2.tariff_gel
         FROM tariff_with_usd t2
         WHERE t2.date = p.date
           AND t2.entity = 'ltd "gardabni thermal power plant"'
         LIMIT 1
        ) AS price_regulated_new_tpp_gel,
        (SELECT t2.tariff_usd
         FROM tariff_with_usd t2
         WHERE t2.date = p.date
           AND t2.entity = 'ltd "gardabni thermal power plant"'
         LIMIT 1
        ) AS price_regulated_new_tpp_usd,

        -- Regulated old TPPs: average of old thermal plants
        (SELECT AVG(t3.tariff_gel)
         FROM tariff_with_usd t3
         WHERE t3.date = p.date
           AND t3.entity IN ('ltd "mtkvari energy"',
                             'ltd "iec" (tbilresi)',
                             'ltd "g power" (capital turbines)')
        ) AS price_regulated_old_tpp_gel,
        (SELECT AVG(t3.tariff_usd)
         FROM tariff_with_usd t3
         WHERE t3.date = p.date
           AND t3.entity IN ('ltd "mtkvari energy"',
                             'ltd "iec" (tbilresi)',
                             'ltd "g power" (capital turbines)')
        ) AS price_regulated_old_tpp_usd

        -- Note: PPA and import prices are NOT available in database
        -- These would need to be estimated or obtained from confidential data

      FROM price_with_usd p
      {price_where}
    )
    SELECT
      p.date,
      p.p_bal_gel AS balancing_price_gel,
      p.p_bal_usd AS balancing_price_usd,
      p.xrate,

      -- Shares
      (s.qty_import / NULLIF(s.total_qty,0)) AS share_import,
      (s.qty_dereg_hydro / NULLIF(s.total_qty,0)) AS share_deregulated_hydro,
      (s.qty_reg_hpp / NULLIF(s.total_qty,0)) AS share_regulated_hpp,
      (s.qty_reg_new_tpp / NULLIF(s.total_qty,0)) AS share_regulated_new_tpp,
      (s.qty_reg_old_tpp / NULLIF(s.total_qty,0)) AS share_regulated_old_tpp,
      (s.qty_ren_ppa / NULLIF(s.total_qty,0)) AS share_renewable_ppa,
      (s.qty_thermal_ppa / NULLIF(s.total_qty,0)) AS share_thermal_ppa,

      -- Reference prices (where available)
      ep.price_deregulated_hydro_gel,
      ep.price_deregulated_hydro_usd,
      ep.price_regulated_hpp_gel,
      ep.price_regulated_hpp_usd,
      ep.price_regulated_new_tpp_gel,
      ep.price_regulated_new_tpp_usd,
      ep.price_regulated_old_tpp_gel,
      ep.price_regulated_old_tpp_usd,

      -- Estimated contributions to balancing price
      -- (share × reference_price) = estimated contribution in GEL/MWh or USD/MWh
      (s.qty_dereg_hydro / NULLIF(s.total_qty,0)) * COALESCE(ep.price_deregulated_hydro_gel, 0)
        AS contribution_deregulated_hydro_gel,
      (s.qty_reg_hpp / NULLIF(s.total_qty,0)) * COALESCE(ep.price_regulated_hpp_gel, 0)
        AS contribution_regulated_hpp_gel,
      (s.qty_reg_new_tpp / NULLIF(s.total_qty,0)) * COALESCE(ep.price_regulated_new_tpp_gel, 0)
        AS contribution_regulated_new_tpp_gel,
      (s.qty_reg_old_tpp / NULLIF(s.total_qty,0)) * COALESCE(ep.price_regulated_old_tpp_gel, 0)
        AS contribution_regulated_old_tpp_gel,
      (s.qty_dereg_hydro / NULLIF(s.total_qty,0)) * COALESCE(ep.price_deregulated_hydro_usd, 0)
        AS contribution_deregulated_hydro_usd,
      (s.qty_reg_hpp / NULLIF(s.total_qty,0)) * COALESCE(ep.price_regulated_hpp_usd, 0)
        AS contribution_regulated_hpp_usd,
      (s.qty_reg_new_tpp / NULLIF(s.total_qty,0)) * COALESCE(ep.price_regulated_new_tpp_usd, 0)
        AS contribution_regulated_new_tpp_usd,
      (s.qty_reg_old_tpp / NULLIF(s.total_qty,0)) * COALESCE(ep.price_regulated_old_tpp_usd, 0)
        AS contribution_regulated_old_tpp_usd,

      -- Sum of known contributions
      COALESCE((s.qty_dereg_hydro / NULLIF(s.total_qty,0)) * ep.price_deregulated_hydro_gel, 0) +
      COALESCE((s.qty_reg_hpp / NULLIF(s.total_qty,0)) * ep.price_regulated_hpp_gel, 0) +
      COALESCE((s.qty_reg_new_tpp / NULLIF(s.total_qty,0)) * ep.price_regulated_new_tpp_gel, 0) +
      COALESCE((s.qty_reg_old_tpp / NULLIF(s.total_qty,0)) * ep.price_regulated_old_tpp_gel, 0)
        AS total_known_contributions_gel,
      COALESCE((s.qty_dereg_hydro / NULLIF(s.total_qty,0)) * ep.price_deregulated_hydro_usd, 0) +
      COALESCE((s.qty_reg_hpp / NULLIF(s.total_qty,0)) * ep.price_regulated_hpp_usd, 0) +
      COALESCE((s.qty_reg_new_tpp / NULLIF(s.total_qty,0)) * ep.price_regulated_new_tpp_usd, 0) +
      COALESCE((s.qty_reg_old_tpp / NULLIF(s.total_qty,0)) * ep.price_regulated_old_tpp_usd, 0)
        AS total_known_contributions_usd,

      -- Residual (PPA + import contribution, not directly observable)
      p.p_bal_gel - (
        COALESCE((s.qty_dereg_hydro / NULLIF(s.total_qty,0)) * ep.price_deregulated_hydro_gel, 0) +
        COALESCE((s.qty_reg_hpp / NULLIF(s.total_qty,0)) * ep.price_regulated_hpp_gel, 0) +
        COALESCE((s.qty_reg_new_tpp / NULLIF(s.total_qty,0)) * ep.price_regulated_new_tpp_gel, 0) +
        COALESCE((s.qty_reg_old_tpp / NULLIF(s.total_qty,0)) * ep.price_regulated_old_tpp_gel, 0)
      ) AS residual_contribution_ppa_import_gel,
      p.p_bal_usd - (
        COALESCE((s.qty_dereg_hydro / NULLIF(s.total_qty,0)) * ep.price_deregulated_hydro_usd, 0) +
        COALESCE((s.qty_reg_hpp / NULLIF(s.total_qty,0)) * ep.price_regulated_hpp_usd, 0) +
        COALESCE((s.qty_reg_new_tpp / NULLIF(s.total_qty,0)) * ep.price_regulated_new_tpp_usd, 0) +
        COALESCE((s.qty_reg_old_tpp / NULLIF(s.total_qty,0)) * ep.price_regulated_old_tpp_usd, 0)
      ) AS residual_contribution_ppa_import_usd,

      -- Shares of PPA and import (for context on residual)
      (s.qty_ren_ppa / NULLIF(s.total_qty,0)) +
      (s.qty_thermal_ppa / NULLIF(s.total_qty,0)) +
      (s.qty_import / NULLIF(s.total_qty,0)) AS share_ppa_import_total

    FROM (
      SELECT *
      FROM price_with_usd p
      {price_where}
    ) p
    LEFT JOIN shares s ON s.date = p.date
    LEFT JOIN entity_prices ep ON ep.date = p.date
    ORDER BY p.date
    """
    # Add LIMIT using configured MAX_ROWS
    sql_with_limit = f"{sql.strip()}\nLIMIT {MAX_ROWS};"
    res = conn.execute(text(sql_with_limit), params)
    df = pd.DataFrame(res.fetchall(), columns=list(res.keys()))
    check_dataframe_memory(df)
    return df
