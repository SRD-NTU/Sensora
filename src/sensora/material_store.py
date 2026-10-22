"""Add and delete matrix and filler records in the application database.

Records are never edited in place. Supabase writes are transactional and recorded in
the database change history. Local writes are preceded by a timestamped SQLite backup.
"""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import re
import sqlite3

from . import database
from .supabase_store import SupabaseError, configured_store

PREFIX = {"matrix": "M", "filler": "F"}
NUMERIC = {
    "youngs_modulus_pa": ("Young's modulus", 0.0, math.inf),
    "poisson_ratio": ("Poisson ratio", -1.0, 0.5),
    "relative_permittivity": ("Relative permittivity", 0.0, math.inf),
    "density_kg_m3": ("Density", 0.0, math.inf),
}


class MaterialStoreError(ValueError):
    """The requested insertion or deletion is invalid."""


def _path(path: Path | None) -> Path:
    return Path(path or database.DEFAULT_DB).resolve()


def next_material_id(role: str, path: Path | None = None) -> str:
    """Return the next free ID for a role: one more than the highest existing number."""
    if role not in PREFIX:
        raise MaterialStoreError("Role must be 'matrix' or 'filler'.")
    prefix = PREFIX[role]
    # Supabase inserts allocate their ID atomically inside the database function.
    ids = [row["material_id"] for row in database.read_database(path).material_candidates
           if row["role"] == role]
    numbers = [int(m.group(1)) for value in ids if (m := re.fullmatch(rf"{prefix}(\d+)", value))]
    return f"{prefix}{max(numbers, default=0) + 1:03d}"


def _backup(path: Path, backup_dir: Path | None) -> Path:
    folder = Path(backup_dir) if backup_dir else path.parent / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}{path.suffix}"
    source, destination = sqlite3.connect(path), sqlite3.connect(target)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return target


def add_material(
    role: str,
    values: dict,
    *,
    path: Path | None = None,
    backup_dir: Path | None = None,
) -> str:
    """Insert one matrix or filler with an automatically assigned ID and return that ID."""
    if role not in PREFIX:
        raise MaterialStoreError("Role must be 'matrix' or 'filler'.")
    name = str(values.get("name") or "").strip()
    if not name:
        raise MaterialStoreError("Name is required.")
    numbers = {}
    for key, (label, low, high) in NUMERIC.items():
        try:
            number = float(values[key])
        except (KeyError, TypeError, ValueError):
            raise MaterialStoreError(f"{label} is required and must be a number.") from None
        if not math.isfinite(number) or not low < number < high:
            bound = f"greater than {low:g}" + (f" and less than {high:g}" if math.isfinite(high) else "")
            raise MaterialStoreError(f"{label} must be {bound}.")
        numbers[key] = number

    record = {
        "name": name,
        "role": role,
        "formulation_or_phase": str(values.get("formulation_or_phase") or "").strip(),
        **{key: format(number, "g") for key, number in numbers.items()},
        "evidence_status": "provisional",
        "limitations": str(values.get("limitations") or "").strip(),
    }
    try:
        if path is None and (store := configured_store()) is not None:
            return store.add_record(record)
    except SupabaseError as exc:
        raise MaterialStoreError(str(exc)) from None
    path = _path(path)
    material_id = next_material_id(role, path)
    record["material_id"] = material_id
    _backup(path, backup_dir)
    connection = sqlite3.connect(path)
    try:
        with connection:
            duplicate = connection.execute(
                "SELECT 1 FROM material_candidates WHERE role = ? AND lower(name) = lower(?)",
                (role, name),
            ).fetchone()
            if duplicate:
                raise MaterialStoreError(f"A {role} named '{name}' already exists.")
            connection.execute(
                "INSERT INTO material_candidates VALUES (?, ?, ?, ?)",
                (material_id, role, name, json.dumps(record, sort_keys=True)),
            )
    finally:
        connection.close()
    return material_id


def delete_materials(
    material_ids: list[str],
    *,
    path: Path | None = None,
    backup_dir: Path | None = None,
) -> dict[str, int]:
    """Delete materials by ID."""
    ids = sorted(set(material_ids))
    if not ids:
        raise MaterialStoreError("Select at least one material to delete.")
    try:
        if path is None and (store := configured_store()) is not None:
            return store.delete_records(ids)
    except SupabaseError as exc:
        raise MaterialStoreError(str(exc)) from None
    path = _path(path)
    marks = ", ".join("?" for _ in ids)
    _backup(path, backup_dir)
    connection = sqlite3.connect(path)
    try:
        with connection:
            found = {row[0] for row in connection.execute(
                f"SELECT material_id FROM material_candidates WHERE material_id IN ({marks})", ids
            )}
            if found != set(ids):
                raise MaterialStoreError(
                    "Unknown material: " + ", ".join(sorted(set(ids) - found)) + "."
                )
            deleted = connection.execute(
                f"DELETE FROM material_candidates WHERE material_id IN ({marks})", ids
            ).rowcount
    finally:
        connection.close()
    return {"materials": deleted}
