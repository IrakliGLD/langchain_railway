-- F10 B4.B — production-safe READ-ONLY database checks (2026-07-19)
-- Run in the PRODUCTION Supabase SQL editor (project qvmqmmcglqmhachqaezt).
-- Strictly read-only: only SELECTs against pg_catalog / information_schema.
-- No DDL, no DML, no mutations. Archive each result set as B4.B DB evidence.

-- 1) public tables + RLS status (RLS must be ON for user-owned tables).
select c.relname          as table_name,
       c.relrowsecurity   as rls_enabled,
       c.relforcerowsecurity as rls_forced
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public' and c.relkind = 'r'
order by c.relname;

-- 2) RLS policies per table (who can do what).
select schemaname, tablename, policyname, cmd, roles, qual is not null as has_using, with_check is not null as has_check
from pg_policies
where schemaname = 'public'
order by tablename, policyname;

-- 3) SECURITY DEFINER functions in public (the protected RPCs) + their owners.
select p.proname                          as function_name,
       pg_get_function_identity_arguments(p.oid) as args,
       p.prosecdef                        as security_definer,
       pg_get_userbyid(p.proowner)        as owner
from pg_proc p
join pg_namespace n on n.oid = p.pronamespace
where n.nspname = 'public'
order by p.proname;

-- 4) materialized views present.
select matviewname from pg_matviews where schemaname = 'public' order by matviewname;

-- 5) explicit table/view privileges granted to the Supabase roles.
--    Verify anon/authenticated do NOT hold broad direct DML on protected data;
--    protected access should be mediated by SECURITY DEFINER RPCs.
select table_name, grantee, string_agg(privilege_type, ', ' order by privilege_type) as privileges
from information_schema.role_table_grants
where table_schema = 'public'
  and grantee in ('anon', 'authenticated', 'service_role')
group by table_name, grantee
order by table_name, grantee;

-- 6) EXECUTE grants on the protected RPCs (who may call them).
select r.routine_name, g.grantee, g.privilege_type
from information_schema.routine_privileges g
join information_schema.routines r
  on r.specific_name = g.specific_name and r.specific_schema = g.specific_schema
where g.specific_schema = 'public'
  and g.grantee in ('anon', 'authenticated', 'service_role')
order by r.routine_name, g.grantee;

-- 7) key integrity constraints (e.g. chat_history JSONB shape checks) exist.
select conrelid::regclass::text as table_name, conname, contype,
       pg_get_constraintdef(oid) as definition
from pg_constraint
where connamespace = 'public'::regnamespace
  and contype in ('c', 'f', 'p', 'u')
order by table_name, conname;

-- 8) indexes on the admin-listing / high-traffic tables.
select tablename, indexname, indexdef
from pg_indexes
where schemaname = 'public'
order by tablename, indexname;

-- 9) dedicated runtime role exists, is NOINHERIT, and defaults to read-only.
select rolname, rolinherit, rolcanlogin, rolbypassrls
from pg_roles
where rolname = 'enai_api_readonly';

-- 10) quarantine/reconciliation health (P6.B): should be 0 for a clean prod
--     (also gates the B5.B P6.B compatibility removal). Guarded so it does not
--     error if P6.B has not been applied to production yet.
select case
         when to_regclass('public.chat_history_jsonb_quarantine') is null
           then -1  -- P6.B quarantine table not present
         else (select count(*) from public.chat_history_jsonb_quarantine)
       end as chat_history_jsonb_quarantine_rows;

select case
         when to_regclass('public.chat_history') is null then -1
         else (
           select count(*) from public.chat_history
           where (chart_data is not null and jsonb_typeof(chart_data) not in ('array','object','null'))
              or (chart_metadata is not null and jsonb_typeof(chart_metadata) not in ('object','null'))
         )
       end as chat_history_legacy_string_rows;
