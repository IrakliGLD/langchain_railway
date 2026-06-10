-- Least-privilege read-only role for the Enai analyst API (audit S7).
--
-- WHY: the backend connects to Postgres via a direct connection string
-- (SUPABASE_DB_URL), NOT via PostgREST. Row-Level Security on the
-- anon/authenticated roles therefore does NOT gate this connection — whatever
-- role is embedded in the URL governs access. Today that is a broadly
-- privileged role, so the ONLY thing preventing writes or reads of
-- non-whitelisted tables (e.g. auth.users) is the app-layer table whitelist
-- plus the per-transaction `SET TRANSACTION READ ONLY`. This migration moves
-- that boundary into the database: a role whose sole privilege is SELECT on the
-- whitelisted analytical relations. With it in place, the app-layer whitelist
-- and the read-only transaction become genuine defense-in-depth.
--
-- HOW TO APPLY (Supabase):
--   1. Open the project's SQL editor (or use your migration tooling).
--   2. Replace CHANGE_ME_STRONG_SECRET with a strong generated password.
--   3. Run this script.
--   4. Point SUPABASE_DB_URL at this role:
--        postgresql://enai_api_readonly:<password>@<host>:<port>/<db>
--   5. Verify with the checks at the bottom of this file.
--
-- The ingestion path (ingest_all_documents.py, which WRITES to the knowledge
-- schema) must use a SEPARATE, write-capable role/connection — never this one.

-- 1. Role (idempotent). Store the password only in SUPABASE_DB_URL.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'enai_api_readonly') THEN
    CREATE ROLE enai_api_readonly LOGIN PASSWORD 'CHANGE_ME_STRONG_SECRET';
  END IF;
END
$$;

-- 2. Strip inherited/default privileges, then grant only what the request path needs.
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM enai_api_readonly;
REVOKE ALL ON SCHEMA public FROM enai_api_readonly;

GRANT CONNECT ON DATABASE postgres TO enai_api_readonly;  -- adjust if the DB name is not "postgres"
GRANT USAGE ON SCHEMA public TO enai_api_readonly;

-- 3. SELECT on exactly the whitelisted analytical relations.
--    This list MUST stay in sync with config.STATIC_ALLOWED_TABLES
--    (enforced by tests/test_config.py::test_readonly_role_grants_match_whitelist).
GRANT SELECT ON
    public.dates_mv,
    public.entities_mv,
    public.price_with_usd,
    public.tariff_with_usd,
    public.tech_quantity_view,
    public.trade_derived_entities,
    public.monthly_cpi_mv,
    public.energy_balance_long_mv,
    public.mv_balancing_trade_with_tariff
TO enai_api_readonly;

-- 4. Vector knowledge retrieval (read path) reads these two tables in the
--    `knowledge` schema (knowledge/vector_store.py). Grant read-only access.
GRANT USAGE ON SCHEMA knowledge TO enai_api_readonly;
GRANT SELECT ON knowledge.documents, knowledge.document_chunks TO enai_api_readonly;

-- 5. Deliberately NOT granted: any write privilege, sequence usage, or access to
--    other schemas (auth, storage, extensions, vault, ...). Do not add them here.

-- NOTE on views: standard Postgres views run with owner privileges
-- (security_invoker = off, the default), so SELECT on the view/matview above is
-- sufficient and the role does NOT need access to their base tables. If any of
-- these relations is later recreated as a `security_invoker = on` view, this
-- role will also need SELECT on that view's base tables — re-verify after any
-- such change.

-- ---------------------------------------------------------------------------
-- VERIFICATION (run while connected AS enai_api_readonly):
--   SELECT 1 FROM public.price_with_usd LIMIT 1;   -- expect: succeeds
--   SELECT 1 FROM knowledge.document_chunks LIMIT 1;-- expect: succeeds
--   SELECT 1 FROM auth.users LIMIT 1;              -- expect: permission denied
--   CREATE TEMP TABLE t(x int);                    -- expect: permission denied
--   INSERT INTO public.price_with_usd DEFAULT VALUES; -- expect: permission denied
-- ---------------------------------------------------------------------------
