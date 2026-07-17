-- P7/F8 read-only inventory of inherited and PUBLIC database privileges.
-- Safe to run in the Supabase SQL editor. This file changes no database state.
-- Export every result grid before preparing any revoke/replacement-grant change.

select
  current_database() as database_name,
  current_user as connected_role,
  current_setting('default_transaction_read_only', true) as default_transaction_read_only,
  now() as observed_at;

select
  rolname, rolsuper, rolinherit, rolcreaterole, rolcreatedb, rolcanlogin,
  rolreplication, rolbypassrls
from pg_roles
where rolname in (
  'enai_api_readonly', 'anon', 'authenticated', 'service_role',
  'authenticator', 'supabase_auth_admin', 'supabase_storage_admin', 'postgres'
)
order by rolname;

with recursive inherited_roles as (
  select oid, rolname, 0 as depth
  from pg_roles
  where rolname = 'enai_api_readonly'
  union
  select parent.oid, parent.rolname, child.depth + 1
  from inherited_roles child
  join pg_auth_members membership on membership.member = child.oid
  join pg_roles parent on parent.oid = membership.roleid
)
select rolname, depth
from inherited_roles
order by depth, rolname;

select
  database_row.datname as database_name,
  coalesce(grantee.rolname, 'PUBLIC') as grantee,
  privilege.privilege_type,
  privilege.is_grantable
from pg_database database_row
cross join lateral aclexplode(
  coalesce(database_row.datacl, acldefault('d', database_row.datdba))
) privilege
left join pg_roles grantee on grantee.oid = privilege.grantee
where database_row.datname = current_database()
order by grantee, privilege.privilege_type;

select
  namespace.nspname as schema_name,
  coalesce(grantee.rolname, 'PUBLIC') as grantee,
  privilege.privilege_type,
  privilege.is_grantable
from pg_namespace namespace
cross join lateral aclexplode(
  coalesce(namespace.nspacl, acldefault('n', namespace.nspowner))
) privilege
left join pg_roles grantee on grantee.oid = privilege.grantee
where namespace.nspname not like 'pg\_%' escape '\'
  and namespace.nspname <> 'information_schema'
order by namespace.nspname, grantee, privilege.privilege_type;

select table_schema, table_name, grantee, privilege_type, is_grantable
from information_schema.table_privileges
where grantee in (
  'PUBLIC', 'enai_api_readonly', 'anon', 'authenticated', 'service_role',
  'authenticator', 'supabase_auth_admin', 'supabase_storage_admin'
)
order by table_schema, table_name, grantee, privilege_type;

select object_schema, object_name, object_type, grantee, privilege_type, is_grantable
from information_schema.usage_privileges
where grantee in (
  'PUBLIC', 'enai_api_readonly', 'anon', 'authenticated', 'service_role',
  'authenticator', 'supabase_auth_admin', 'supabase_storage_admin'
)
order by object_schema, object_name, grantee, privilege_type;

select routine_schema, routine_name, specific_name, grantee, privilege_type, is_grantable
from information_schema.routine_privileges
where grantee in (
  'PUBLIC', 'enai_api_readonly', 'anon', 'authenticated', 'service_role',
  'authenticator', 'supabase_auth_admin', 'supabase_storage_admin'
)
order by routine_schema, routine_name, grantee, privilege_type;

select
  owner.rolname as object_owner,
  namespace.nspname as schema_name,
  default_acl.defaclobjtype as object_type,
  coalesce(grantee.rolname, 'PUBLIC') as grantee,
  privilege.privilege_type,
  privilege.is_grantable
from pg_default_acl default_acl
join pg_roles owner on owner.oid = default_acl.defaclrole
left join pg_namespace namespace on namespace.oid = default_acl.defaclnamespace
cross join lateral aclexplode(default_acl.defaclacl) privilege
left join pg_roles grantee on grantee.oid = privilege.grantee
order by owner.rolname, namespace.nspname nulls first, grantee, privilege.privilege_type;