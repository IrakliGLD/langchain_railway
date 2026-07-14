# P5.A Database Gateway Activation Runbook

**Date:** 2026-07-14
**Scope:** Backend repository only

## Purpose

This package routes runtime database connections through one classified circuit-breaker gateway. Fallback SQL, typed tools, vector operations, analyzer/pipeline enrichment, readiness probes, and schema reflection now share the same breaker decision and outcome recording.

The frontend/Supabase application remains independent. This package changes no frontend source, Edge Function, database schema, grant, or shared HTTP contract.

## Failure classification

- Connection failures, SQLAlchemy pool timeouts, serialization failures, deadlocks, statement timeouts, database shutdown, resource exhaustion, and system failures count toward the database breaker.
- SQL syntax, missing relation/column, constraint, validation, and content errors do not increment the breaker.
- A non-transient SQL error releases a half-open probe as an infrastructure success because it proves the database is reachable.
- When the breaker is open, every guarded runtime DB path fails before acquiring a connection.

## Environment and migration requirements

No new environment variable or database migration is required. Keep the existing database URL, pool, breaker threshold, and breaker reset settings.

## Deploy

1. Deploy the backend commit containing this runbook to staging.
2. Do not deploy or copy frontend/Supabase files for this package.
3. Confirm `/healthz` remains live and `/readyz` reports ready with the normal database connection.
4. Exercise one typed data query and one vector-backed knowledge query.
5. Promote the same immutable backend commit to production after the staging checks pass.

## Failure-injection checks

Run these only in an isolated staging environment:

1. Cause a SQL syntax/schema failure through a direct internal test or staging diagnostic and confirm the database breaker failure count does not increase.
2. Block or invalidate the staging database connection and repeat a guarded readiness/query call until the configured failure threshold is reached.
3. Confirm the breaker opens and subsequent fallback SQL, typed-tool, vector, enrichment, and schema-reflection paths do not reach the database.
4. Restore connectivity, wait for the configured reset interval, and confirm exactly one half-open probe runs and a success closes the breaker.
5. Confirm logs contain only the operation label and exception type, not SQL text, parameters, credentials, or raw database errors.

## Rollback

Roll back only the backend deployment to the prior immutable commit. No frontend rollback, Edge deployment, environment rollback, or database rollback is required.

## Remaining P5 work

P5.2 does not change connection-pool capacity or add request-deadline-aware pool checkout/cancellation. Those remain P5.3 work. Provider breaker ownership remains P5.4.
