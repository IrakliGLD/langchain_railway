"""Non-destructive P7.A acceptance probes for the deployed runtime role.

Run this with the runtime ``SUPABASE_DB_URL``.  Every mutation/DDL probe is
wrapped in a transaction and rolled back even if an over-privileged role lets
it succeed; success is nevertheless reported as a failed security assertion.
"""

from __future__ import annotations

import os
import sys

import psycopg

EXPECTED_ROLE = os.getenv("ENAI_DB_RUNTIME_ROLE", "enai_api_readonly").strip()


def _psycopg_url(raw: str) -> str:
    return raw.replace("postgresql+psycopg://", "postgresql://", 1)


def _must_fail(conn: psycopg.Connection, statement: str, label: str) -> str | None:
    # Use an explicit transaction so the probe is rolled back on both the
    # expected-error and unexpected-success paths.  A context-managed
    # transaction would commit an unexpectedly successful DDL/DML probe before
    # the caller had a chance to report the privilege failure.
    conn.rollback()
    try:
        conn.execute("begin")
        conn.execute(statement)
    except Exception:
        conn.rollback()
        return None
    finally:
        conn.rollback()
    return f"{label} unexpectedly succeeded"


def main() -> int:
    database_url = os.getenv("SUPABASE_DB_URL", "").strip()
    if not database_url:
        print("SUPABASE_DB_URL is required", file=sys.stderr)
        return 2

    failures: list[str] = []
    with psycopg.connect(_psycopg_url(database_url)) as conn:
        role, read_only = conn.execute(
            "select current_user, current_setting('default_transaction_read_only')"
        ).fetchone()
        if role != EXPECTED_ROLE:
            failures.append(f"expected role {EXPECTED_ROLE!r}, got {role!r}")
        if read_only != "on":
            failures.append("default_transaction_read_only is not on")

        conn.execute("select 1 from public.price_with_usd limit 0")
        conn.execute("select 1 from knowledge.document_chunks limit 0")
        conn.rollback()

        for statement, label in [
            ("select 1 from auth.users limit 0", "auth.users read"),
            ("update knowledge.documents set title = title where false", "knowledge write"),
            ("create temp table enai_p7_denial_probe(x integer)", "DDL"),
            ("set role postgres", "role escalation"),
        ]:
            failure = _must_fail(conn, statement, label)
            if failure:
                failures.append(failure)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("PASS: runtime role identity, allowed reads, and denial probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
