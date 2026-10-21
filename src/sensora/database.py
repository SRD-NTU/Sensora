"""SQLite storage for the matrix and filler records."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from .config import ROOT

DEFAULT_DB = ROOT / "data/materials.sqlite3"

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS material_candidates ("
    "material_id TEXT PRIMARY KEY, role TEXT NOT NULL CHECK(role IN ('matrix', 'filler')), "
    "name TEXT NOT NULL, record_json TEXT NOT NULL)"
)


@dataclass(frozen=True)
class MaterialDatabase:
    material_candidates: list[dict]

    def fingerprint(self) -> str:
        payload = json.dumps(self.material_candidates, sort_keys=True)
        return sha256(payload.encode()).hexdigest()


def read_database(path: Path | None = None) -> MaterialDatabase:
    path = Path(path or DEFAULT_DB).resolve()
    if not path.is_file():
        raise ValueError("The materials database is not installed.")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT record_json FROM material_candidates ORDER BY material_id"
        ).fetchall()
    return MaterialDatabase(material_candidates=[json.loads(row[0]) for row in rows])
