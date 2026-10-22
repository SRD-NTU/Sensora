"""Knowledge passages the assistant retrieves from: documents, reviewed devices, and live data."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


from .config import ROOT
from .database import MaterialDatabase

KNOWLEDGE_FILE = ROOT / "data/knowledge/knowledge.jsonl"
MAX_CHARS = 1100


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str
    source: str
    kind: str


def split_markdown(text: str, doc_title: str, source: str, prefix: str) -> list[Chunk]:
    """Split a markdown document into passages at headings, then at paragraph breaks."""
    sections: list[tuple[str, list[str]]] = [(doc_title, [])]
    for line in text.splitlines():
        if line.startswith(">"):                      # historical banners
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)", line)
        if heading:
            title = heading.group(2).strip()
            sections.append((title if heading.group(1) == "#" else f"{doc_title} > {title}", []))
        else:
            sections[-1][1].append(line)
    chunks: list[Chunk] = []
    for title, lines in sections:
        body = "\n".join(lines).strip()
        if len(body) < 60:
            continue
        pieces, current = [], ""
        for paragraph in re.split(r"\n\s*\n", body):
            if current and len(current) + len(paragraph) > MAX_CHARS:
                pieces.append(current)
                current = ""
            current = f"{current}\n\n{paragraph}".strip()
        if current:
            pieces.append(current)
        for number, piece in enumerate(pieces, start=1):
            chunks.append(Chunk(f"{prefix}#{len(chunks) + 1}", title, piece, source, "document"))
    return chunks


def load_knowledge(path: Path | None = None) -> list[Chunk]:
    path = Path(path or KNOWLEDGE_FILE)
    if not path.is_file():
        return []
    return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _pa(value: float) -> str:
    return f"{value / 1000:,.4g} kPa"


def material_chunks(database: MaterialDatabase) -> list[Chunk]:
    """One passage per material plus ranked overviews, always in step with the database."""
    chunks, matrices, fillers = [], [], []
    for row in database.material_candidates:
        modulus, nu = float(row["youngs_modulus_pa"]), float(row["poisson_ratio"])
        eps, density = float(row["relative_permittivity"]), float(row["density_kg_m3"])
        (matrices if row["role"] == "matrix" else fillers).append((row["name"], modulus, eps))
        text = (
            f"{row['name']} is a {row['role']} in the Sensora database (ID {row['material_id']}). "
            f"Form: {row.get('formulation_or_phase', '') or 'not recorded'}. "
            f"Young's modulus {_pa(modulus)}; Poisson ratio {nu:g}; relative permittivity {eps:g}; "
            f"density {density:g} kg/m3. Evidence status: {row.get('evidence_status', 'unknown').replace('_', ' ')}. "
            f"Caveats: {row.get('limitations', '').strip() or 'none recorded'}."
        )
        chunks.append(Chunk(f"material:{row['material_id']}", row["name"], text,
                            f"Sensora materials database, {row['material_id']}", "material"))
    if matrices:
        ordered = sorted(matrices, key=lambda item: item[1])
        text = (
            "Matrices from softest to stiffest, with the approximate pressure at which the unfilled matrix "
            "reaches the model's 10% strain limit: "
            + "; ".join(f"{n} (E {_pa(e)}, limit about {e * 0.10 / 1000:,.4g} kPa, permittivity {p:g})"
                        for n, e, p in ordered) + ". Softer matrices respond more to pressure but cap the usable pressure range."
        )
        chunks.append(Chunk("material:matrices", "Matrix comparison", text,
                            "Sensora materials database", "material"))
    if fillers:
        ordered = sorted(fillers, key=lambda item: item[2], reverse=True)
        text = (
            "Fillers from highest to lowest relative permittivity: "
            + "; ".join(f"{n} (permittivity {p:g}, E {_pa(e)})" for n, e, p in ordered)
            + ". A higher permittivity raises the composite's permittivity more at the same volume fraction."
        )
        chunks.append(Chunk("material:fillers", "Filler comparison", text,
                            "Sensora materials database", "material"))
    return chunks


def session_chunks(result, database: MaterialDatabase) -> list[Chunk]:
    """Passages describing the user's most recent optimizer run and its experiment plan."""
    from .composite_physics import Targets
    from .experiment_plan import build_experiment_plan

    if result is None or result.table.empty:
        return []
    report = result.report
    low = report["targets"]["pressure_min_kpa"]
    high = report["targets"]["pressure_max_kpa"]
    settings = report["settings"]
    source = f"Latest Sensora optimizer run ({low:g} to {high:g} kPa)"
    top = result.table.head(5)
    lines = [
        f"#{row['rank']} {row['formulation']}: robust score {row['robust_fitness']:.5f}, "
        f"effective permittivity {row['effective_relative_permittivity']:.2f}, "
        f"effective modulus {row['effective_youngs_modulus_mpa'] * 1000:,.4g} kPa, "
        f"worst-case strain {row['worst_case_maximum_strain_percent']:.2f}%"
        for _, row in top.iterrows()
    ]
    overview = (
        f"For a requested pressure range of {low:g} to {high:g} kPa the genetic algorithm "
        f"(population {settings['population_size']}, {settings['generations']} iterations, matrices "
        f"{settings['minimum_active_matrices']} to {settings['maximum_active_matrices']}, fillers "
        f"{settings['minimum_active_fillers']} to {settings['maximum_active_fillers']}) scored "
        f"{report['unique_formulations_evaluated']} formulations, of which "
        f"{report['feasible_formulations_evaluated']} stayed under the 10% strain limit. "
        f"Filler loading mode: {report.get('loading_regime', 'dilute')}. "
        "Top results: " + " | ".join(lines) + "."
    )
    best = result.table.iloc[0]
    detail = (
        f"The best formulation is {best['formulation']}. Nominal values: effective permittivity "
        f"{best['effective_relative_permittivity']:.3f}, effective Young's modulus "
        f"{best['effective_youngs_modulus_mpa'] * 1000:,.4g} kPa, Poisson ratio "
        f"{best['effective_poisson_ratio']:.3f}, interval-average sensitivity "
        f"{best['nominal_average_delta_c_over_c0_sensitivity_per_kpa']:.5f} per kPa. Conservative "
        f"stress-test values used for the score: permittivity {best['robust_relative_permittivity']:.3f}, "
        f"modulus {best['robust_youngs_modulus_mpa'] * 1000:,.4g} kPa, sensitivity "
        f"{best['robust_average_delta_c_over_c0_sensitivity_per_kpa']:.5f} per kPa. Worst-case strain "
        f"{best['worst_case_maximum_strain_percent']:.2f}% against the 10% limit."
    )
    chunks = [
        Chunk("session:overview", "Latest optimizer result", overview, source, "session"),
        Chunk("session:best", "Best formulation details", detail, source, "session"),
    ]
    try:
        plan = build_experiment_plan(result, database, Targets(low, high))
        rows = [
            f"{r.experiment_id} {r.purpose}: {r.formulation}, predicted permittivity "
            f"{r.predicted_relative_permittivity:.2f}, predicted sensitivity "
            f"{r.predicted_sensitivity_per_kpa:.5f} per kPa"
            for r in plan.table.itertuples()
        ]
        chunks.append(Chunk(
            "session:plan", "Latest experiment plan",
            "Experiment plan samples: " + "; ".join(rows) +
            ". The unfilled control shows whether filler adds a real benefit; the loading series tests the predicted trend.",
            source, "session",
        ))
    except ValueError:
        pass
    return chunks
