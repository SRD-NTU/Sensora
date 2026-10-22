"""Turn an optimization result into a laboratory validation plan.

The plan is deterministic. Every predicted value comes from the same physics model as
the optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import pandas as pd

from .composite_optimization import CompositeOptimizationResult, predict_formulation
from .composite_physics import Targets
from .database import MaterialDatabase

LOADING_SERIES = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class ExperimentPlan:
    table: pd.DataFrame


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
    """Build the sample list and predictions for the top designs."""
    if result.table.empty:
        raise ValueError("There is no optimized formulation to plan experiments for.")
    names = {row["material_id"]: row["name"] for row in database.material_candidates}

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

    return ExperimentPlan(table=table)
