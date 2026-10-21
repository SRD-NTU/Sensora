"""Turn an optimization result into a laboratory validation plan.

The plan is deterministic. Every predicted value comes from the same physics model as
the optimizer, and every material caveat comes from the database record of that material.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from .composite_optimization import CompositeOptimizationResult, predict_formulation
from .composite_physics import Targets
from .database import MaterialDatabase

LOADING_SERIES = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class ExperimentPlan:
    table: pd.DataFrame
    protocol: str


def _formulation_text(matrix_names: dict[str, str], matrix: dict, fillers: dict, names: dict) -> str:
    base = " / ".join(
        (f"{percent:g}% " if len(matrix) > 1 else "") + matrix_names[key] for key, percent in matrix.items()
    )
    if not fillers:
        return base + " (unfilled)"
    return base + " + " + " + ".join(f"{vol:g} vol% {names[key]}" for key, vol in fillers.items())


def build_experiment_plan(
    result: CompositeOptimizationResult,
    database: MaterialDatabase,
    targets: Targets,
) -> ExperimentPlan:
    """Build the sample list, predictions, and measurement protocol for the top designs."""
    if result.table.empty:
        raise ValueError("There is no optimized formulation to plan experiments for.")
    names = {row["material_id"]: row["name"] for row in database.material_candidates}
    records = {row["material_id"]: row for row in database.material_candidates}

    def parse(row) -> tuple[dict, dict]:
        return json.loads(row["matrix_components_json"]), json.loads(row["components_json"])

    samples: list[tuple[str, dict, dict]] = []
    best_matrix, best_fillers = parse(result.table.iloc[0])
    samples.append(("Control: unfilled matrix", best_matrix, {}))
    seen = set()
    for _, row in result.table.iterrows():
        matrix, fillers = parse(row)
        key = (tuple(sorted(matrix)), tuple(sorted(fillers)))
        if key in seen:
            continue
        seen.add(key)
        label = "Top-ranked design" if len(seen) == 1 else f"Ranked design {len(seen)}"
        samples.append((label, matrix, fillers))
        if len(seen) == 3:
            break
    if not result.evaluated.empty:
        others = result.evaluated[result.evaluated.matrix_id != "+".join(best_matrix)]
        if not others.empty:
            alt = others.sort_values("robust_fitness", ascending=False).iloc[0]
            matrix, fillers = parse(alt)
            samples.append(("Alternative matrix", matrix, fillers))
    for share in LOADING_SERIES:
        samples.append((
            f"Loading series ({share:.0%} of top design)", best_matrix,
            {key: round(vol * share, 3) for key, vol in best_fillers.items()},
        ))

    rows = []
    for index, (purpose, matrix, fillers) in enumerate(samples, start=1):
        prediction = predict_formulation(database, matrix, fillers, targets)
        rows.append({
            "experiment_id": f"E{index:02d}",
            "purpose": purpose,
            "formulation": _formulation_text(names, matrix, fillers, names),
            "filler_vol_percent": round(sum(fillers.values()), 3),
            "predicted_relative_permittivity": prediction["relative_permittivity"],
            "predicted_youngs_modulus_kpa": prediction["youngs_modulus_kpa"],
            "predicted_sensitivity_per_kpa": prediction["sensitivity_per_kpa"],
            "predicted_response_at_pmax": prediction["response_at_pmax"],
            "predicted_strain_at_pmax_percent": prediction["strain_at_pmax_percent"],
            "replicates": 3,
        })
    table = pd.DataFrame(rows)

    used = list(dict.fromkeys(
        key for _, matrix, fillers in samples for key in (*matrix, *fillers)
    ))
    caveats = "\n".join(
        f"- **{names[key]}:** {records[key].get('limitations', '').strip() or 'No caveat recorded.'}"
        for key in used
    )
    protocol = f"""# Validation plan for {targets.pressure_min_kpa:g}–{targets.pressure_max_kpa:g} kPa

These designs are model predictions to be tested, not verified results.

## Samples
{len(table)} formulations, each made as **3 independent batches with 3 specimens per batch**
(see the experiment table). The unfilled control and the loading series show whether the
model's trend with filler content is real; the alternative matrix tests the matrix ranking.

## Fixed conditions
Use one electrode area and one starting thickness for every sample, and record both,
because the predicted score assumes a fixed geometry. Record fabrication route, cure,
temperature, and measurement frequency.

## Measurements for every sample
1. **Actual filler loading and density**, to convert mass to volume fraction.
2. **Dispersion and voids** by microscopy; the model assumes separated particles.
3. **Dielectric spectrum and loss** (for example 1 kHz–1 MHz); the model assumes a constant permittivity.
4. **Small-strain compression modulus**, loading and unloading, up to the maximum pressure.
5. **Capacitance versus pressure** over {targets.pressure_min_kpa:g}–{targets.pressure_max_kpa:g} kPa, loading and unloading,
   with repeated cycles; report ΔC/C₀ and the interval-average sensitivity.
6. **Leakage or conductivity**, which matters for semiconducting fillers.

## Material-specific checks (from the database records)
{caveats}

## How to judge the model
- Compare measured with predicted sensitivity. Agreement within about ±25% matches the
  stress-test margin used for provisional materials; larger gaps mean the model misses
  a mechanism, such as aggregation or pressure-dependent permittivity.
- Check whether the measured ranking of formulations matches the predicted ranking.
- Compare every filled sample with the unfilled control from the same batch.
- Keep one formulation untouched by model tuning as an independent test.
"""
    return ExperimentPlan(table=table, protocol=protocol)
