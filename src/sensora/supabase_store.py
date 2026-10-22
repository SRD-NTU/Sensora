"""Server-side access to Supabase's Data API, using only the Python standard library."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
import tomllib
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from .config import ROOT


class SupabaseError(ValueError):
    """An actionable storage error that is safe to display without credentials."""


@dataclass(frozen=True)
class SupabaseStore:
    url: str
    secret_key: str = field(repr=False)

    def request(self, resource: str, *, payload=None, query: dict | None = None):
        endpoint = f"{self.url}/rest/v1/{resource}"
        if query:
            endpoint += "?" + urlencode(query)
        request = Request(
            endpoint,
            data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
            headers={
                "apikey": self.secret_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=20) as response:
                return json.load(response)
        except HTTPError as exc:
            try:
                error = json.load(exc)
                code = error.get("code", "") if isinstance(error, dict) else ""
            except (ValueError, OSError):
                code = ""
            if code in {"PGRST202", "PGRST205", "42P01", "42883"}:
                message = "Supabase materials storage is not set up. Run supabase/setup.sql in the SQL Editor."
            elif code == "23505":
                message = "A material with this name or ID already exists."
            elif code == "P0002":
                message = "A selected material no longer exists. Refresh the page and select again."
            elif exc.code in {401, 403}:
                message = "Supabase access was denied. Check the server secret key and database permissions."
            elif code in {"22023", "23514", "23502"}:
                message = "Supabase rejected invalid material data. Check the required fields."
            else:
                message = f"Supabase request failed (HTTP {exc.code}). Please try again."
            raise SupabaseError(message) from None
        except (URLError, TimeoutError, OSError):
            # Do not retry mutations: the server may have committed before a timeout.
            raise SupabaseError(
                "Could not reach Supabase. Check the connection and refresh before retrying any change."
            ) from None
        except ValueError:
            raise SupabaseError("Supabase returned an invalid response.") from None

    def read_records(self) -> list[dict]:
        records = []
        last_id = None
        while True:
            query = {"select": "material_id,record_json", "order": "material_id.asc", "limit": 500}
            if last_id is not None:
                query["material_id"] = f"gt.{last_id}"
            page = self.request("material_candidates", query=query)
            if not isinstance(page, list):
                raise SupabaseError("Supabase returned an invalid materials table.")
            if not page:
                return records
            for row in page:
                if (not isinstance(row, dict) or not isinstance(row.get("record_json"), dict)
                        or not isinstance(row.get("material_id"), str)):
                    raise SupabaseError("Supabase returned an invalid material record.")
                records.append(row["record_json"])
            next_id = page[-1]["material_id"]
            if next_id == last_id:
                raise SupabaseError("Supabase did not advance through the materials table.")
            last_id = next_id

    def add_record(self, record: dict) -> str:
        material_id = self.request("rpc/sensora_add_material", payload={"p_record": record})
        if not isinstance(material_id, str) or not re.fullmatch(r"[MF]\d+", material_id):
            raise SupabaseError("Supabase returned an invalid material ID. Refresh before retrying.")
        return material_id

    def delete_records(self, material_ids: list[str]) -> dict[str, int]:
        count = self.request("rpc/sensora_delete_materials", payload={"p_ids": material_ids})
        if type(count) is not int or count != len(set(material_ids)):
            raise SupabaseError("Supabase returned an unexpected deletion count. Refresh before retrying.")
        return {"materials": count}


def configured_store() -> SupabaseStore | None:
    """Use Supabase when configured; incomplete configuration never falls back to SQLite."""
    values = {key: os.environ.get(key, "").strip() for key in ("SUPABASE_URL", "SUPABASE_SECRET_KEY")}
    secrets_path = ROOT / ".streamlit/secrets.toml"
    if not all(values.values()) and secrets_path.is_file():
        try:
            with secrets_path.open("rb") as handle:
                secrets = tomllib.load(handle)
        except (OSError, ValueError):
            raise SupabaseError("Could not read .streamlit/secrets.toml. Check its TOML formatting.") from None
        for key in values:
            if not values[key]:
                values[key] = str(secrets.get(key, "")).strip()
    if not any(values.values()):
        return None
    if not all(values.values()):
        raise SupabaseError("Set both SUPABASE_URL and SUPABASE_SECRET_KEY in the server secrets.")
    url = values["SUPABASE_URL"].rstrip("/")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path):
        raise SupabaseError("SUPABASE_URL must be the HTTPS project URL, without a path or credentials.")
    if not values["SUPABASE_SECRET_KEY"].startswith("sb_secret_"):
        raise SupabaseError("SUPABASE_SECRET_KEY must be a server secret key starting with sb_secret_.")
    return SupabaseStore(url, values["SUPABASE_SECRET_KEY"])
