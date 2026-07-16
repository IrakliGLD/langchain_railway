# F5 global database coordination activation and verification

**Date:** 2026-07-16
**Scope:** P5.3 backend-only global database admission, readiness reservation, and secondary-work ownership
**Runtime constraint:** one backend worker and one Railway replica until P5.6 external shared state is complete

## Implemented contract

- All runtime database paths continue to use `core/db_gateway.py`; there is no second connection path.
- One process-wide `DatabaseWorkCoordinator` owns application and control admission. Application work cannot consume the slot reserved for schema-readiness and runtime-identity probes.
- Primary requests and parallel secondary evidence share the same application admission budget. The SQLAlchemy pool ceiling is separately bounded and must be at least the coordinator ceiling.
- Queue admission, connection checkout, and PostgreSQL execution all consume the same parent request deadline. Work is rejected before checkout when cleanup and connection reserves no longer fit.
- PostgreSQL receives a transaction-local `statement_timeout` equal to the smaller of the configured DB timeout and remaining request time minus cleanup allowance. This is the supported cancellation mechanism for timed-out SQL; the least-privilege API role is not granted `pg_cancel_backend` or backend-termination rights.
- Stage 0.8 uses one process-wide bounded executor and bounded pending queue. Every worker inherits an actor-bound request identity and a child deadline that cannot outlive its parent.
- Parent exit cancels queued secondary futures and waits only the configured cleanup interval. A non-cooperative overrun is counted as `secondary_orphaned` and logged without prompts, SQL, actor IDs, or other private content.
- Aggregate coordinator state is available under the already protected `/metrics` response at `resilience.db_work`.

No SQL migration, Supabase Edge deployment, or frontend deployment is required for F5.

## Capacity variables and defaults

| Railway variable | Default | Constraint/purpose |
|---|---:|---|
| `ENAI_DB_POOL_SIZE` | `3` | SQLAlchemy persistent client pool. |
| `ENAI_DB_MAX_OVERFLOW` | `2` | Temporary SQLAlchemy clients; pool ceiling is size plus overflow. |
| `ENAI_DB_MAX_CONCURRENCY` | `4` | Total admitted DB work; must not exceed the pool ceiling. |
| `ENAI_DB_CONTROL_RESERVED_SLOTS` | `1` | Reserved readiness/control capacity; at least one. |
| `ENAI_DB_QUEUE_TIMEOUT_MS` | `2000` | Maximum admission wait, further reduced by request remaining time. |
| `ENAI_DB_SECONDARY_WORKERS` | `2` | Process-wide Stage 0.8 workers; cannot exceed application capacity. |
| `ENAI_DB_SECONDARY_PENDING_LIMIT` | `4` | Running plus queued secondary futures. |
| `ENAI_DB_SECONDARY_DRAIN_TIMEOUT_MS` | `250` | Parent-exit cleanup tolerance for secondary work. |
| `ENAI_DB_POOL_TIMEOUT_SECONDS` | `2` | SQLAlchemy checkout bound. |
| `ENAI_DB_CONNECT_TIMEOUT_SECONDS` | `3` | PostgreSQL connection bound. |
| `ENAI_DB_STATEMENT_TIMEOUT_MS` | `30000` | Upper bound for each transaction; the request deadline may reduce it. |

Derived application capacity is:

```text
application capacity = ENAI_DB_MAX_CONCURRENCY - ENAI_DB_CONTROL_RESERVED_SLOTS
pool ceiling         = ENAI_DB_POOL_SIZE + ENAI_DB_MAX_OVERFLOW
```

With defaults, application capacity is 3, control capacity is 1, total admitted work is 4, and the physical client-pool ceiling is 5.

Startup fails closed if the pool cannot leave at least one application and one control connection, if total coordinator concurrency exceeds the pool ceiling, or if worker/pending values are inconsistent.

## Choose production capacity from Supabase/Supavisor

Do not increase the defaults merely because Railway has spare CPU. The limiting resource is the Supabase/Supavisor allocation.

1. In the Supabase dashboard, open **Project Settings -> Database** and record the transaction-pool/Supavisor pool allocation for the connection endpoint used by `SUPABASE_DB_URL`.
2. In the Supabase SQL editor, record current connection pressure without changing data:

```sql
select setting::integer as max_connections
from pg_settings
where name = 'max_connections';

select coalesce(usename, '<internal>') as role_name,
       state,
       count(*) as connections
from pg_stat_activity
group by coalesce(usename, '<internal>'), state
order by connections desc;
```

3. Reserve capacity for Supabase platform services, migrations/admin access, Auth/Storage/Realtime, and another operator connection. Do not assign the whole project limit to this app.
4. Because the backend is restricted to one replica, assign that replica a conservative client budget. Set the SQLAlchemy pool ceiling no higher than that budget.
5. Keep `ENAI_DB_MAX_CONCURRENCY` below the pool ceiling when possible. Retain one control slot; use the remainder as application capacity.
6. Start with the defaults unless measured capacity is lower. Increase by one only after a canary shows low queue rejection, stable database latency, and no readiness failures.
7. Record the dashboard allocation, calculation, variables, replica count, date, and approver in the deployment evidence. This measurement is the remaining capacity-acceptance gate; source code cannot infer a hosted pool allocation safely.

## Automated evidence

The local suite covers:

- normal admission and release;
- application saturation with a still-available control lane;
- bounded slow/queued work;
- simulated SQLAlchemy pool exhaustion;
- open database breaker behavior before admission;
- request-deadline expiry while queued;
- PostgreSQL statement-timeout calculation;
- cancellation of queued secondary work and bounded draining of running work;
- explicit detection of an uncooperative secondary overrun;
- simultaneous primary and secondary work under one application ceiling;
- actor/request identity and child-deadline propagation into secondary workers;
- aggregate, content-free coordinator metrics; and
- a source regression rejecting reintroduction of per-request thread pools or DB-gateway bypasses.

## Deployment and canary

1. Deploy the F5 backend commit to Railway. Do not change frontend, Edge, or Supabase SQL.
2. Keep one Uvicorn worker and one Railway replica.
3. If the defaults fit the measured Supavisor allocation, deploy without new variables. Otherwise set the capacity variables above in Railway and redeploy.
4. Confirm `/readyz` returns 200 before traffic cutover.
5. Send one ordinary data request and one request that requires at least two secondary evidence sources.
6. Run a short bounded concurrency canary through the normal browser/Edge route. Include normal requests, deliberately slow queries, and simultaneous primary/secondary evidence requests. Do not bypass the gateway or expose backend secrets to a load client.
7. During the canary, inspect the protected `/metrics` endpoint over a restricted/admin path. Do not expose it publicly merely for this test. Check `resilience.db_work`:
   - `peak_application <= application_capacity`;
   - `peak_control <= control_capacity`;
   - `active_application` and `active_control` return to `0` after the run;
   - `secondary_outstanding` returns to `0`;
   - `secondary_orphaned` remains `0` for production tool calls;
   - queue waits remain below `queue_timeout_seconds`; and
   - `/readyz` continues to return 200 while application capacity is saturated.
8. Open the DB breaker in a controlled staging/failure-injection environment and confirm DB paths reject without increasing coordinator admission. Do not deliberately disrupt the only live production database.
9. Abort/navigate away from a browser request and verify backend work terminates by its absolute deadline; after cleanup, `secondary_outstanding` must be zero.
10. Record request IDs only. Do not capture prompts, responses, SQL text, actor IDs, or access tokens in the evidence package.

## Exit evidence and blockers

Local implementation is complete when the focused and broad test suites pass. Production F5 acceptance additionally requires the operator to record:

- measured Supabase/Supavisor capacity and the selected Railway values;
- one-replica/one-worker confirmation;
- normal, saturation, pool-pressure, breaker-open, readiness, cancellation, and simultaneous primary/secondary canary results;
- protected `resilience.db_work` before/peak/after snapshots; and
- zero production `secondary_outstanding` and `secondary_orphaned` after cleanup.

A generic Python thread cannot be force-killed safely. F5 therefore cancels queued futures, bounds parent cleanup, uses PostgreSQL `statement_timeout` for actual DB cancellation, and treats any surviving non-cooperative task as a measured release blocker rather than hiding it. Do not mark the live exit gate complete if `secondary_orphaned` increases.

## Rollback

Roll back the backend commit and restore the previous Railway variable set. No database or frontend rollback is needed. Keep one replica throughout rollback. If the only failure is excessive queue rejection, first restore the defaults; do not raise concurrency until Supavisor capacity has been re-measured.