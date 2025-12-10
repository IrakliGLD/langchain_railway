"""
SQL generation, validation, and sanitization.

Handles:
- SQL sanitization (removing comments, fences)
- Table whitelist validation using AST parsing
- Synonym resolution and auto-correction
- LIMIT clause enforcement
- CTE (Common Table Expression) handling
"""
import re
import logging
from typing import Set

from fastapi import HTTPException
from sqlglot import parse_one, exp
from sqlglot.errors import ParseError

from config import (
    ALLOWED_TABLES,
    TABLE_SYNONYMS,
    SYNONYM_PATTERNS,
    MAX_ROWS,
    LIMIT_PATTERN
)

log = logging.getLogger("Enai")


def simple_table_whitelist_check(sql: str) -> None:
    """
    CRITICAL Pre-parsing safety check using a robust SQL parser.

    Extracts all table references from the AST for whitelisting.
    Handles CTEs (Common Table Expressions) properly by excluding them
    from whitelist validation.

    Args:
        sql: SQL query to validate

    Raises:
        HTTPException 400: If query contains unauthorized tables or invalid SQL

    Examples:
        >>> simple_table_whitelist_check("SELECT * FROM price_with_usd LIMIT 10")
        # Returns None if valid

        >>> simple_table_whitelist_check("SELECT * FROM unauthorized_table")
        # Raises HTTPException
    """
    cleaned_tables: Set[str] = set()

    try:
        parsed_expression = parse_one(sql, read='bigquery')

        # --- FIX: 1. Extract CTE names ---
        cte_names: Set[str] = set()
        with_clause = parsed_expression.find(exp.With)
        if with_clause:
            for cte in with_clause.expressions:
                if cte.alias is not None:  # Explicit None check for anonymous CTEs
                    cte_names.add(cte.alias.lower())
        # ---------------------------------

        # 2. Traverse the AST to find all table expressions
        for table_exp in parsed_expression.find_all(exp.Table):

            t_raw = table_exp.name.lower()
            t_name = t_raw.split('.')[0]

            # --- FIX: 2. Skip CTE names from whitelisting ---
            if t_name in cte_names:
                continue
            # ---------------------------------------------

            # Apply synonym mapping and perform the strict whitelist check
            t_canonical = TABLE_SYNONYMS.get(t_name, t_name)

            if t_canonical in ALLOWED_TABLES:
                cleaned_tables.add(t_canonical)
            else:
                # Re-raise the exception with the specific name that failed the check
                raise HTTPException(
                    status_code=400,
                    detail=f"❌ Unauthorized table or view: `{t_name}`. Allowed: {sorted(ALLOWED_TABLES)}"
                )

    except ParseError as e:
        # If the SQL is too broken to parse (e.g., truly invalid SQL), reject it.
        # For security, any unparseable query should be rejected.
        log.error(f"SQL PARSE ERROR: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"❌ SQL Validation Error (Parse Failed): The query could not be reliably parsed for security review. Details: {e}"
        )
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        log.error(f"Unexpected error during SQL parsing: {e}")
        # Reject on any other unexpected error
        raise HTTPException(
            status_code=400,
            detail=f"❌ SQL Validation Error (Unexpected): An unexpected error occurred during security review."
        )

    if not cleaned_tables:
        # This handles valid queries that might not have a FROM clause (e.g., SELECT 1)
        # or where the FROM clause is in a subquery/CTE that the parser handles,
        # but the logic above didn't capture (unlikely with find_all(exp.Table)).
        log.warning("⚠️ No tables were extracted. Allowing flow for statements without a FROM (e.g. SELECT 1).")
        return

    log.info(f"✅ Pre-validation passed. Tables: {list(cleaned_tables)}")
    return


def sanitize_sql(sql: str) -> str:
    """
    Basic sanitization: strip comments and fences.

    Removes:
    - Markdown code fences (```)
    - Single-line SQL comments (--)
    - Leading/trailing whitespace

    Security:
    - Only allows SELECT statements

    Args:
        sql: Raw SQL query

    Returns:
        Sanitized SQL query

    Raises:
        HTTPException 400: If SQL is not a SELECT statement

    Examples:
        >>> sanitize_sql("```sql\\nSELECT * FROM dates_mv\\n```")
        'SELECT * FROM dates_mv'

        >>> sanitize_sql("SELECT * FROM dates_mv -- comment")
        'SELECT * FROM dates_mv'
    """
    # Remove markdown fences and initial/trailing whitespace
    sql = sql.strip().strip('`').strip()

    # Remove single-line comments
    sql = re.sub(r"--.*", "", sql)

    # Basic protection against non-SELECT statements
    if not sql.lower().startswith("select"):
        raise HTTPException(400, "Only SELECT statements are allowed.")

    return sql


def plan_validate_repair(sql: str) -> str:
    """
    Repair phase: Auto-corrects common table/view synonyms and ensures a LIMIT.

    Table whitelisting occurs BEFORE this function is called.

    Repairs:
    1. Replaces table synonyms with canonical names (using pre-compiled regex)
    2. Appends LIMIT if missing
    3. Removes trailing semicolons before adding LIMIT

    Args:
        sql: SQL query to repair

    Returns:
        Repaired SQL query with synonyms resolved and LIMIT enforced

    Examples:
        >>> plan_validate_repair("SELECT * FROM prices WHERE year=2023")
        'SELECT * FROM price_with_usd WHERE year=2023\\nLIMIT 5000'

        >>> plan_validate_repair("SELECT * FROM dates_mv LIMIT 10")
        'SELECT * FROM dates_mv LIMIT 10'  # Already has LIMIT
    """
    _sql = sql

    # Phase 1: Repair synonyms using pre-compiled regex patterns (optimized)
    try:
        for pattern, replacement in SYNONYM_PATTERNS:
            _sql = pattern.sub(replacement, _sql)
    except Exception as e:
        log.warning(f"⚠️ Synonym auto-correction failed: {e}")
        # Not a critical failure, continue with original SQL

    # Phase 2: Append LIMIT if missing (using pre-compiled pattern)
    if " from " in _sql.lower() and not LIMIT_PATTERN.search(_sql):

        # CRITICAL FIX: Remove the trailing semicolon if it exists
        _sql = _sql.rstrip().rstrip(';')

        # Append LIMIT without a preceding semicolon
        _sql = f"{_sql}\nLIMIT {MAX_ROWS}"

    return _sql
