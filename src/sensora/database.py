"""Read matrix and filler records from Supabase."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from .supabase_store import configured_store


@dataclass(frozen=True)
class MaterialDatabase:
    material_candidates: list[dict]

    def fingerprint(self) -> str:
        payload = json.dumps(self.material_candidates, sort_keys=True)
        return sha256(payload.encode()).hexdigest()


def read_database() -> MaterialDatabase:
    return MaterialDatabase(material_candidates=configured_store().read_records())
