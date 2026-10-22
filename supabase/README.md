# Supabase setup

Sensora uses Supabase's HTTPS Data API from the Python server. It needs Python 3.11+
and adds no Python dependencies. The public browser never receives the secret key.

1. Run `supabase/setup.sql` in your project's Supabase SQL Editor. It creates the
   schema and seeds the initial 20 materials in one transaction. Re-running it
   preserves matching records and aborts if a seeded record conflicts.
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the
   placeholder with a new `sb_secret_...` key. The real secrets file is ignored by Git.
   On Streamlit Community Cloud, put these same two values in the app's Secrets
   settings. Other hosts can supply `SUPABASE_URL` and `SUPABASE_SECRET_KEY` as
   environment variables, which override the project secrets file.
3. Restart Sensora. The Database tab should list the seeded materials and say
   that materials are stored in Supabase.

Use the project URL `https://akzfcdwemgeaipydnflp.supabase.co`. The publishable key
is not needed by this server-based app. An API key can read and change existing
data, but initial SQL setup requires the SQL Editor or an authorized Supabase
management connection. A Postgres password or `DATABASE_URL` is not required
for this integration.

Sensora requires both Supabase settings. Missing or invalid settings show a clear
error. All material reads, additions, and deletions use Supabase.

The `material_candidates` table retains all material properties and existing IDs.
Server-only SQL functions allocate new IDs and delete selections atomically. RLS
is enabled and anonymous/authenticated API roles have no access to these tables or
functions. The server's secret key uses `service_role`. Sensora's existing UI allows
anyone who can open the app to add and delete materials, so host the app privately
for trusted users unless you add application authentication.

`sensora_material_history` keeps the complete old and new records for every insert,
update, and delete, including imports. It supports recovering deleted records through
the SQL Editor; it is not an off-site backup.

References: [API keys](https://supabase.com/docs/guides/getting-started/api-keys),
[Data API](https://supabase.com/docs/guides/api),
[Database functions](https://supabase.com/docs/guides/database/functions).
