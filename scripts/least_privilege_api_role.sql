-- P7.A least-privilege read-only role for the Enai analytics backend.
--
-- Apply this as a database owner/admin.  The role is deliberately created
-- NOLOGIN so no placeholder password can accidentally become a credential.
-- After this script succeeds, generate a strong password outside source
-- control and run this one statement manually in the Supabase SQL editor:
--
--   ALTER ROLE enai_api_readonly LOGIN PASSWORD '<generated secret>';
--
-- Put the resulting connection URL only in Railway SUPABASE_DB_URL.  Set
-- ENAI_DB_RUNTIME_ROLE=enai_api_readonly (the production default) and retain a
-- separate write-capable credential for offline knowledge ingestion.

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'enai_api_readonly') THEN
    CREATE ROLE enai_api_readonly
      NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
  END IF;
END
$$;

ALTER ROLE enai_api_readonly NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE enai_api_readonly CONNECTION LIMIT 5;
ALTER ROLE enai_api_readonly SET default_transaction_read_only = on;
ALTER ROLE enai_api_readonly SET statement_timeout = '30s';
ALTER ROLE enai_api_readonly SET lock_timeout = '5s';
ALTER ROLE enai_api_readonly SET idle_in_transaction_session_timeout = '30s';

REVOKE ALL ON DATABASE postgres FROM enai_api_readonly;
GRANT CONNECT ON DATABASE postgres TO enai_api_readonly;

REVOKE ALL ON ALL TABLES IN SCHEMA public FROM enai_api_readonly;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM enai_api_readonly;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM enai_api_readonly;
REVOKE ALL ON SCHEMA public FROM enai_api_readonly;
GRANT USAGE ON SCHEMA public TO enai_api_readonly;

-- This list MUST stay byte-for-byte aligned with config.STATIC_ALLOWED_TABLES;
-- tests/test_config.py enforces the relation set.
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

REVOKE ALL ON ALL TABLES IN SCHEMA knowledge FROM enai_api_readonly;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA knowledge FROM enai_api_readonly;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA knowledge FROM enai_api_readonly;
REVOKE ALL ON SCHEMA knowledge FROM enai_api_readonly;
GRANT USAGE ON SCHEMA knowledge TO enai_api_readonly;
GRANT SELECT ON knowledge.documents, knowledge.document_chunks TO enai_api_readonly;

-- PostgreSQL PUBLIC privileges are inherited by every role and cannot be
-- overridden with a per-role deny.  Do not globally revoke PUBLIC function or
-- schema grants from this backend migration: Supabase/PostgREST consumers may
-- rely on them.  Inventory and revoke/re-grant those privileges using the P7.A
-- runbook across both independently deployed applications before attestation.

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'auth') THEN
    EXECUTE 'REVOKE ALL ON SCHEMA auth FROM enai_api_readonly';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'storage') THEN
    EXECUTE 'REVOKE ALL ON SCHEMA storage FROM enai_api_readonly';
  END IF;
END
$$;
