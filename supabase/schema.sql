-- Used by scripts/prepare_supabase.py to build the one-time setup file.
-- Only the server's service_role can access Sensora records or write functions.
create table if not exists public.material_candidates (
    material_id text primary key,
    role text not null check (role in ('matrix', 'filler')),
    name text not null check (name = btrim(name) and length(name) > 0),
    record_json jsonb not null,
    constraint sensora_material_id check (
        (role = 'matrix' and material_id ~ '^M[0-9]+$') or
        (role = 'filler' and material_id ~ '^F[0-9]+$')
    ),
    constraint sensora_record_identity check (
        jsonb_typeof(record_json) = 'object' and
        (record_json->>'material_id') is not distinct from material_id and
        (record_json->>'role') is not distinct from role and
        (record_json->>'name') is not distinct from name
    ),
    constraint sensora_numeric_properties check (coalesce(
        (record_json->>'youngs_modulus_pa')::double precision > 0 and
        (record_json->>'youngs_modulus_pa')::double precision < 'Infinity'::double precision and
        (record_json->>'poisson_ratio')::double precision > -1 and
        (record_json->>'poisson_ratio')::double precision < 0.5 and
        (record_json->>'relative_permittivity')::double precision > 0 and
        (record_json->>'relative_permittivity')::double precision < 'Infinity'::double precision and
        (record_json->>'density_kg_m3')::double precision > 0 and
        (record_json->>'density_kg_m3')::double precision < 'Infinity'::double precision,
        false
    ))
);

create unique index if not exists sensora_material_name
    on public.material_candidates (role, lower(name));

create table if not exists public.sensora_material_history (
    change_id bigint generated always as identity primary key,
    changed_at timestamptz not null default now(),
    operation text not null check (operation in ('INSERT', 'UPDATE', 'DELETE')),
    material_id text not null,
    old_record jsonb,
    new_record jsonb
);

alter table public.material_candidates enable row level security;
alter table public.sensora_material_history enable row level security;
revoke all on public.material_candidates from public, anon, authenticated, service_role;
revoke all on public.sensora_material_history from public, anon, authenticated, service_role;
grant select, insert, delete on public.material_candidates to service_role;
grant select on public.sensora_material_history to service_role;

create or replace function public.sensora_record_material_change()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
    insert into public.sensora_material_history (operation, material_id, old_record, new_record)
    values (TG_OP, coalesce(NEW.material_id, OLD.material_id), OLD.record_json, NEW.record_json);
    return null;
end;
$$;

revoke all on function public.sensora_record_material_change() from public, anon, authenticated;
create or replace trigger sensora_material_change
    after insert or update or delete on public.material_candidates
    for each row execute function public.sensora_record_material_change();

create or replace function public.sensora_add_material(p_record jsonb)
returns text
language plpgsql
security invoker
set search_path = ''
as $$
declare
    material_role text := p_record->>'role';
    material_name text := btrim(p_record->>'name');
    prefix text;
    next_number bigint;
    new_id text;
begin
    if material_role is null or material_role not in ('matrix', 'filler') then
        raise exception 'Role must be matrix or filler.' using errcode = '22023';
    end if;
    if material_name is null or material_name = '' then
        raise exception 'Name is required.' using errcode = '22023';
    end if;

    -- Serialize inserts and deletions so simultaneous app sessions cannot reuse an ID.
    lock table public.material_candidates in share row exclusive mode;
    prefix := case material_role when 'matrix' then 'M' else 'F' end;
    select coalesce(max(substring(material_id from 2)::bigint), 0) + 1
        into next_number from public.material_candidates where role = material_role;
    new_id := prefix || lpad(next_number::text, greatest(3, length(next_number::text)), '0');
    insert into public.material_candidates (material_id, role, name, record_json)
    values (new_id, material_role, material_name,
        p_record || jsonb_build_object('material_id', new_id, 'name', material_name));
    return new_id;
end;
$$;

create or replace function public.sensora_delete_materials(p_ids text[])
returns integer
language plpgsql
security invoker
set search_path = ''
as $$
declare
    requested integer;
    found integer;
    deleted integer;
begin
    select count(distinct id) into requested from unnest(p_ids) as ids(id);
    if requested = 0 or array_position(p_ids, null) is not null then
        raise exception 'Select at least one valid material.' using errcode = '22023';
    end if;
    lock table public.material_candidates in share row exclusive mode;
    select count(*) into found from public.material_candidates where material_id = any(p_ids);
    if found <> requested then
        raise exception 'A selected material no longer exists.' using errcode = 'P0002';
    end if;
    delete from public.material_candidates where material_id = any(p_ids);
    get diagnostics deleted = row_count;
    return deleted;
end;
$$;

revoke all on function public.sensora_add_material(jsonb) from public, anon, authenticated;
revoke all on function public.sensora_delete_materials(text[]) from public, anon, authenticated;
grant execute on function public.sensora_add_material(jsonb) to service_role;
grant execute on function public.sensora_delete_materials(text[]) to service_role;

