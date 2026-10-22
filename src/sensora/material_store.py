"""Validate material changes and store them in Supabase."""
from __future__ import annotations

import math

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


def add_material(role: str, values: dict) -> str:
    """Insert a material; Supabase atomically assigns its ID."""
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
        return configured_store().add_record(record)
    except SupabaseError as exc:
        raise MaterialStoreError(str(exc)) from None


def delete_materials(material_ids: list[str]) -> dict[str, int]:
    """Delete selected materials by ID through Supabase's transactional function."""
    ids = sorted(set(material_ids))
    if not ids:
        raise MaterialStoreError("Select at least one material to delete.")
    try:
        return configured_store().delete_records(ids)
    except SupabaseError as exc:
        raise MaterialStoreError(str(exc)) from None
