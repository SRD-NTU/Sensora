"""Reproducible genetic optimization of dilute dielectric composites."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

import numpy as np
import pandas as pd

from .composite_physics import (
    Targets,
    bruggeman_permittivity,
    lewis_nielsen_modulus,
    maxwell_garnett_multiphase,
    mori_tanaka_elasticity,
    planar_capacitance_response,
    self_consistent_elasticity,
)
from .database import MaterialDatabase


MATRIX_UNITS = 10  # matrix-phase blend resolution: 10 vol% of the matrix phase per unit


@dataclass(frozen=True)
class GeneticOptimizationSettings:
    population_size: int = 120
    generations: int = 60
    elite_count: int = 12
    tournament_size: int = 3
    mutation_probability: float = 0.30
    loading_step_vol_percent: float = 0.1
    maximum_total_filler_vol_percent: float = 4.9
    minimum_total_filler_vol_percent: float = 0.0
    loading_regime: str = "dilute"
    minimum_active_fillers: int = 1
    maximum_active_fillers: int = 4
    minimum_active_matrices: int = 1
    maximum_active_matrices: int = 1
    result_count: int = 20
    random_seed: int = 42

    def validate(self) -> None:
        if self.population_size < 2 or self.generations < 1:
            raise ValueError("GA population must be at least 2 and generations must be positive.")
        if self.elite_count < 1:
            raise ValueError("GA elite count must be at least 1.")
        if not 1 <= self.minimum_active_fillers <= self.maximum_active_fillers:
            raise ValueError("Filler count limits must satisfy 1 <= minimum <= maximum.")
        if not 1 <= self.minimum_active_matrices <= self.maximum_active_matrices <= MATRIX_UNITS:
            raise ValueError("Matrix count limits must satisfy 1 <= minimum <= maximum.")
        if not 0 < self.loading_step_vol_percent < 5:
            raise ValueError("The loading step must lie between zero and 5 vol%.")
        if self.loading_regime == "dilute":
            if not 0 < self.maximum_total_filler_vol_percent < 5:
                raise ValueError("Total filler must remain below 5 vol% in the dilute regime.")
            if self.minimum_total_filler_vol_percent != 0:
                raise ValueError("The dilute regime has no minimum total filler.")
        elif self.loading_regime == "concentrated":
            if not 5 <= self.minimum_total_filler_vol_percent <= self.maximum_total_filler_vol_percent <= 20:
                raise ValueError("The concentrated regime needs 5 <= minimum <= maximum <= 20 vol% total filler.")
        else:
            raise ValueError("Loading regime must be 'dilute' or 'concentrated'.")


@dataclass(frozen=True)
class CompositeOptimizationResult:
    table: pd.DataFrame
    curves: pd.DataFrame
    history: pd.DataFrame
    report: dict
    evaluated: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class _Limits:
    matrices: int
    fillers: int
    min_matrices: int
    max_matrices: int
    min_fillers: int
    max_fillers: int
    maximum_units: int
    minimum_units: int = 0


def _number(row: dict, key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Material {row.get('material_id', '?')} has no valid {key}.") from exc
    if not np.isfinite(value):
        raise ValueError(f"Material {row.get('material_id', '?')} has no valid {key}.")
    return value


def _uncertainty(row: dict) -> tuple[float, float, float]:
    """Return transparent stress-test margins for E, epsilon_r, and Poisson ratio."""
    if row.get("evidence_status") == "screening_ready":
        return 0.10, 0.10, 0.02
    return 0.25, 0.25, 0.05


def _random_genome(rng: np.random.Generator, limits: _Limits) -> tuple[int, ...]:
    count = int(rng.integers(limits.min_matrices, limits.max_matrices + 1))
    chosen = rng.choice(limits.matrices, size=count, replace=False)
    blend = np.zeros(limits.matrices, dtype=int)
    blend[chosen] = 1 + rng.multinomial(MATRIX_UNITS - count, np.full(count, 1 / count))

    active = int(rng.integers(limits.min_fillers, limits.max_fillers + 1))
    filler_indices = rng.choice(limits.fillers, size=active, replace=False)
    total = int(rng.integers(max(active, limits.minimum_units), limits.maximum_units + 1))
    loads = np.ones(active, dtype=int)
    if total > active:
        loads += rng.multinomial(total - active, np.full(active, 1 / active))
    genes = np.zeros(limits.fillers, dtype=int)
    genes[filler_indices] = loads
    return (*blend.tolist(), *genes.tolist())


def _repair_matrices(block: np.ndarray, limits: _Limits) -> np.ndarray:
    units = np.maximum(block, 0).astype(int)
    active = np.flatnonzero(units)
    if len(active) > limits.max_matrices:
        keep = active[np.argsort(units[active], kind="stable")[-limits.max_matrices:]]
        mask = np.ones(len(units), dtype=bool)
        mask[keep] = False
        units[mask] = 0
    if units.sum() == 0:
        units[0] = MATRIX_UNITS
    while units.sum() > MATRIX_UNITS:
        units[int(np.argmax(units))] -= 1
    while units.sum() < MATRIX_UNITS:
        units[int(np.argmax(units))] += 1
    while np.count_nonzero(units) < limits.min_matrices:
        donor = int(np.argmax(units))
        if units[donor] <= 1:
            break
        units[donor] -= 1
        units[int(np.flatnonzero(units == 0)[0])] += 1
    return units


def _repair_fillers(block: np.ndarray, limits: _Limits) -> np.ndarray:
    loads = np.maximum(block, 0).astype(int)
    active = np.flatnonzero(loads)
    if len(active) > limits.max_fillers:
        keep = active[np.argsort(loads[active], kind="stable")[-limits.max_fillers:]]
        mask = np.ones(len(loads), dtype=bool)
        mask[keep] = False
        loads[mask] = 0
    if loads.sum() == 0:
        loads[0] = 1
    while np.count_nonzero(loads) < limits.min_fillers:
        loads[int(np.flatnonzero(loads == 0)[0])] = 1
    while loads.sum() < limits.minimum_units:
        loads[int(np.argmax(loads))] += 1
    while loads.sum() > limits.maximum_units:
        loads[int(np.argmax(loads))] -= 1
    return loads


def _repair(genome, limits: _Limits) -> tuple[int, ...]:
    genes = np.asarray(genome, dtype=int)
    blend = _repair_matrices(genes[: limits.matrices], limits)
    loads = _repair_fillers(genes[limits.matrices:], limits)
    return (*blend.tolist(), *loads.tolist())


def _mutate(genome, rng: np.random.Generator, limits: _Limits) -> tuple[int, ...]:
    genes = np.asarray(genome, dtype=int).copy()
    blend, loads = genes[: limits.matrices], genes[limits.matrices:]
    if rng.random() < 0.25:
        if limits.max_matrices == 1 or rng.random() < 0.5:
            blend[:] = 0
            blend[int(rng.integers(limits.matrices))] = MATRIX_UNITS
        else:
            source = int(rng.integers(limits.matrices))
            target = int(rng.integers(limits.matrices))
            if blend[source] > 0:
                amount = int(rng.integers(1, blend[source] + 1))
                blend[source] -= amount
                blend[target] += amount
    if rng.random() < 0.35:
        source = int(rng.integers(len(loads)))
        target = int(rng.integers(len(loads)))
        if loads[source] > 0:
            amount = int(rng.integers(1, loads[source] + 1))
            loads[source] -= amount
            loads[target] += amount
        else:
            loads[target] += int(rng.integers(1, 11))
    if rng.random() < 0.80:
        index = int(rng.integers(len(loads)))
        loads[index] += int(rng.integers(-10, 11))
    return _repair(tuple(genes.tolist()), limits)


def _crossover(first, second, rng: np.random.Generator, limits: _Limits) -> tuple[int, ...]:
    a, b = np.asarray(first, dtype=int), np.asarray(second, dtype=int)
    child = np.where(rng.random(len(a)) < 0.5, a, b)
    donor = a if rng.random() < 0.5 else b
    child[: limits.matrices] = donor[: limits.matrices]  # keep a matrix blend intact
    return _repair(tuple(child.tolist()), limits)


def _blend_matrix(rows: list[dict], weights: np.ndarray) -> dict:
    """Effective matrix of a miscible-blend approximation (an assumption, not a validated law).

    Nominal values use logarithmic mixing for E and epsilon_r. The stress test uses the
    Voigt (stiff) and Reuss (soft) modulus bounds, which reduce exactly to the single-matrix
    margins when one matrix is present.
    """
    e = np.asarray([_number(r, "youngs_modulus_pa") for r in rows])
    nu = np.asarray([_number(r, "poisson_ratio") for r in rows])
    eps = np.asarray([_number(r, "relative_permittivity") for r in rows])
    margins = [_uncertainty(r) for r in rows]
    ue, ueps, unu = (np.asarray([m[i] for m in margins]) for i in range(3))
    if len(rows) == 1:
        return dict(
            e=e[0], nu=nu[0], eps=eps[0], soft_e=e[0] * (1 - ue[0]),
            stiff_e=e[0] * (1 + ue[0]), stiff_nu=float(np.clip(nu[0] - unu[0], -0.95, 0.49)),
            robust_eps=eps[0] * (1 - ueps[0]),
        )
    return dict(
        e=float(np.exp(weights @ np.log(e))),
        nu=float(weights @ nu),
        eps=float(np.exp(weights @ np.log(eps))),
        soft_e=float(1 / np.sum(weights / (e * (1 - ue)))),
        stiff_e=float(weights @ (e * (1 + ue))),
        stiff_nu=float(np.clip(weights @ (nu - unu), -0.95, 0.49)),
        robust_eps=float(np.exp(weights @ np.log(eps * (1 - ueps)))),
    )


def _effective_permittivity(regime: str, matrix_eps, filler_eps, fractions) -> float:
    """Maxwell-Garnett when dilute (below 5 vol%); Bruggeman effective medium when concentrated."""
    rule = bruggeman_permittivity if regime == "concentrated" else maxwell_garnett_multiphase
    return rule(matrix_eps, filler_eps, fractions)


def _effective_elasticity(
    regime: str, matrix_e, matrix_nu, filler_e, filler_nu, fractions, *, poisson: bool = True
) -> tuple[float, float]:
    """Mori-Tanaka when dilute; Lewis-Nielsen modulus with self-consistent Poisson ratio when concentrated."""
    if regime != "concentrated":
        return mori_tanaka_elasticity(matrix_e, matrix_nu, filler_e, filler_nu, fractions)
    modulus = lewis_nielsen_modulus(matrix_e, matrix_nu, filler_e, fractions)
    if not poisson:
        return modulus, float("nan")
    _, ratio = self_consistent_elasticity(matrix_e, matrix_nu, filler_e, filler_nu, fractions)
    return modulus, ratio


def _evaluate(
    genome: tuple[int, ...],
    matrices: list[dict],
    fillers: list[dict],
    targets: Targets,
    step: float,
    regime: str = "dilute",
) -> dict:
    m_count = len(matrices)
    blend_units = genome[:m_count]
    loads = genome[m_count:]
    matrix_indices = [index for index, units in enumerate(blend_units) if units > 0]
    active_matrices = [matrices[index] for index in matrix_indices]
    weights = np.asarray([blend_units[index] / MATRIX_UNITS for index in matrix_indices])
    active_indices = [index for index, units in enumerate(loads) if units > 0]
    active_fillers = [fillers[index] for index in active_indices]
    fractions = np.asarray([loads[index] * step / 100 for index in active_indices])

    matrix = _blend_matrix(active_matrices, weights)
    filler_e = [_number(row, "youngs_modulus_pa") for row in active_fillers]
    filler_nu = [_number(row, "poisson_ratio") for row in active_fillers]
    filler_eps = [_number(row, "relative_permittivity") for row in active_fillers]
    filler_u = [_uncertainty(row) for row in active_fillers]

    effective_eps = _effective_permittivity(regime, matrix["eps"], filler_eps, fractions)
    effective_e, effective_nu = _effective_elasticity(
        regime, matrix["e"], matrix["nu"], filler_e, filler_nu, fractions
    )
    soft_e, _ = _effective_elasticity(
        regime,
        matrix["soft_e"],
        matrix["nu"],
        [value * (1 - uncertainty[0]) for value, uncertainty in zip(filler_e, filler_u, strict=True)],
        filler_nu,
        fractions,
        poisson=False,
    )
    worst_strain = targets.pressure_max_kpa * 1000 / soft_e
    feasible = worst_strain <= 0.10 + 1e-12

    robust_eps = _effective_permittivity(
        regime,
        matrix["robust_eps"],
        [value * (1 - uncertainty[1]) for value, uncertainty in zip(filler_eps, filler_u, strict=True)],
        fractions,
    )
    robust_e, robust_nu = _effective_elasticity(
        regime,
        matrix["stiff_e"],
        matrix["stiff_nu"],
        [value * (1 + uncertainty[0]) for value, uncertainty in zip(filler_e, filler_u, strict=True)],
        [float(np.clip(value - uncertainty[2], -0.95, 0.49)) for value, uncertainty in zip(filler_nu, filler_u, strict=True)],
        fractions,
    )

    pressure_endpoints = np.asarray([targets.pressure_min_kpa, targets.pressure_max_kpa]) * 1000
    interval = targets.pressure_max_kpa - targets.pressure_min_kpa
    nominal_sensitivity = np.nan
    robust_sensitivity = np.nan
    nominal_geometry_sensitivity = np.nan
    robust_score = -1e12 - worst_strain
    nominal_endpoint_response = np.nan
    if feasible:
        nominal_response = planar_capacitance_response(
            pressure_endpoints, effective_e, effective_nu, boundary="free_lateral"
        )
        robust_response = planar_capacitance_response(
            pressure_endpoints, robust_e, robust_nu, boundary="free_lateral"
        )
        nominal_sensitivity = float((nominal_response[1] - nominal_response[0]) / interval)
        robust_sensitivity = float((robust_response[1] - robust_response[0]) / interval)
        nominal_geometry_sensitivity = effective_eps * nominal_sensitivity
        robust_score = robust_eps * robust_sensitivity
        nominal_endpoint_response = float(nominal_response[1])

    components = {
        fillers[index]["material_id"]: round(loads[index] * step, 10)
        for index in active_indices
    }
    matrix_components = {
        matrices[index]["material_id"]: blend_units[index] * 100 // MATRIX_UNITS
        for index in matrix_indices
    }
    matrix_label = (
        active_matrices[0]["name"] if len(active_matrices) == 1 else
        " / ".join(f"{weight:.0%} {row['name']}" for weight, row in zip(weights, active_matrices, strict=True))
    )
    formulation = matrix_label + " + " + " + ".join(
        f"{fraction:g} vol% {fillers[index]['name']}"
        for index, fraction in zip(active_indices, components.values(), strict=True)
    )
    return {
        "genome": genome,
        "feasible": feasible,
        "fitness": robust_score,
        "formulation": formulation,
        "matrix_id": "+".join(row["material_id"] for row in active_matrices),
        "matrix": matrix_label,
        "matrix_count": len(active_matrices),
        "matrix_components_json": json.dumps(matrix_components, sort_keys=True),
        "components_json": json.dumps(components, sort_keys=True),
        "filler_count": len(components),
        "total_filler_vol_percent": round(float(sum(components.values())), 10),
        "evidence_status": "provisional" if any(
            row.get("evidence_status") != "screening_ready"
            for row in [*active_matrices, *active_fillers]
        ) else "screening_ready",
        "effective_relative_permittivity": effective_eps,
        "effective_youngs_modulus_mpa": effective_e / 1e6,
        "effective_poisson_ratio": effective_nu,
        "nominal_average_delta_c_over_c0_sensitivity_per_kpa": nominal_sensitivity,
        "nominal_geometry_normalized_capacitance_sensitivity_per_kpa": nominal_geometry_sensitivity,
        "robust_relative_permittivity": robust_eps,
        "robust_youngs_modulus_mpa": robust_e / 1e6,
        "robust_poisson_ratio": robust_nu,
        "robust_average_delta_c_over_c0_sensitivity_per_kpa": robust_sensitivity,
        "robust_fitness": robust_score,
        "predicted_response_at_pmax": nominal_endpoint_response,
        "nominal_maximum_strain_percent": 100 * targets.pressure_max_kpa * 1000 / effective_e,
        "worst_case_maximum_strain_percent": 100 * worst_strain,
    }


def optimize_composites(
    database: MaterialDatabase,
    selected_material_ids: list[str],
    targets: Targets,
    *,
    settings: GeneticOptimizationSettings | None = None,
    curve_points: int = 101,
) -> CompositeOptimizationResult:
    """Optimize matrix blend, filler identities, and filler loadings with a seeded GA."""
    targets.validate()
    settings = settings or GeneticOptimizationSettings()
    settings.validate()
    if curve_points < 2:
        raise ValueError("At least two curve points are required.")
    selected = set(selected_material_ids)
    known = {row["material_id"] for row in database.material_candidates}
    unknown = selected - known
    if unknown:
        raise ValueError(f"Unknown material options: {', '.join(sorted(unknown))}.")
    matrices = [
        row for row in database.material_candidates
        if row["role"] == "matrix" and row["material_id"] in selected
    ]
    fillers = [
        row for row in database.material_candidates
        if row["role"] == "filler" and row["material_id"] in selected
    ]
    if not matrices or not fillers:
        raise ValueError("Allow at least one matrix and one filler.")
    if settings.minimum_active_matrices > len(matrices):
        raise ValueError(
            f"Minimum matrix count {settings.minimum_active_matrices} exceeds the "
            f"{len(matrices)} allowed matrices."
        )
    if settings.minimum_active_fillers > len(fillers):
        raise ValueError(
            f"Minimum filler count {settings.minimum_active_fillers} exceeds the "
            f"{len(fillers)} allowed fillers."
        )

    rng = np.random.default_rng(settings.random_seed)
    limits = _Limits(
        matrices=len(matrices),
        fillers=len(fillers),
        min_matrices=settings.minimum_active_matrices,
        max_matrices=min(settings.maximum_active_matrices, len(matrices)),
        min_fillers=settings.minimum_active_fillers,
        max_fillers=min(settings.maximum_active_fillers, len(fillers)),
        maximum_units=int(round(
            settings.maximum_total_filler_vol_percent / settings.loading_step_vol_percent
        )),
        minimum_units=int(round(
            settings.minimum_total_filler_vol_percent / settings.loading_step_vol_percent
        )),
    )
    population = [_random_genome(rng, limits) for _ in range(settings.population_size)]
    # Seed the bounds for every matrix/filler when the population permits it.
    seeds = []
    for matrix_index in range(len(matrices)):
        for filler_index in range(len(fillers)):
            blend = [0] * len(matrices)
            blend[matrix_index] = MATRIX_UNITS
            loads = [0] * len(fillers)
            loads[filler_index] = limits.maximum_units
            seeds.append(_repair((*blend, *loads), limits))
    population[: min(len(population), len(seeds))] = seeds[: settings.population_size]

    cache: dict[tuple[int, ...], dict] = {}
    history = []
    for generation in range(settings.generations):
        evaluations = []
        for genome in population:
            if genome not in cache:
                cache[genome] = _evaluate(
                    genome, matrices, fillers, targets, settings.loading_step_vol_percent,
                    settings.loading_regime,
                )
            evaluations.append(cache[genome])
        evaluations.sort(key=lambda row: row["fitness"], reverse=True)
        feasible_scores = [row["fitness"] for row in evaluations if row["feasible"]]
        history.append({
            "generation": generation + 1,
            "best_robust_fitness": max(feasible_scores) if feasible_scores else np.nan,
            "mean_feasible_robust_fitness": float(np.mean(feasible_scores)) if feasible_scores else np.nan,
            "feasible_population": len(feasible_scores),
            "unique_formulations_evaluated": len(cache),
        })
        elite_count = min(settings.elite_count, settings.population_size - 1)
        elites = [row["genome"] for row in evaluations[:elite_count]]

        def tournament() -> tuple[int, ...]:
            choices = rng.integers(0, len(evaluations), size=settings.tournament_size)
            return max((evaluations[int(index)] for index in choices), key=lambda row: row["fitness"])["genome"]

        next_population = list(elites)
        while len(next_population) < settings.population_size:
            child = _crossover(tournament(), tournament(), rng, limits)
            if rng.random() < settings.mutation_probability:
                child = _mutate(child, rng, limits)
            next_population.append(child)
        population = next_population

    feasible = sorted(
        (row for row in cache.values() if row["feasible"]),
        key=lambda row: row["fitness"],
        reverse=True,
    )[: settings.result_count]
    table_rows = []
    curves = []
    pressures = np.linspace(targets.pressure_min_kpa, targets.pressure_max_kpa, curve_points)
    for rank, evaluated in enumerate(feasible, start=1):
        row = {key: value for key, value in evaluated.items() if key not in {"genome", "fitness", "feasible"}}
        candidate_id = f"GA-{rank:03d}"
        row = {"rank": rank, "candidate_id": candidate_id, **row}
        table_rows.append(row)
        response = planar_capacitance_response(
            pressures * 1000,
            row["effective_youngs_modulus_mpa"] * 1e6,
            row["effective_poisson_ratio"],
            boundary="free_lateral",
        )
        curves.extend({
            "candidate_id": candidate_id,
            "formulation": row["formulation"],
            "pressure_kpa": float(pressure),
            "predicted_delta_c_over_c0": float(value),
        } for pressure, value in zip(pressures, response, strict=True))

    evaluated = pd.DataFrame([
        {key: row[key] for key in (
            "formulation", "matrix_id", "matrix", "filler_count", "total_filler_vol_percent",
            "effective_relative_permittivity", "effective_youngs_modulus_mpa",
            "effective_poisson_ratio", "robust_fitness",
            "matrix_components_json", "components_json",
        )}
        for row in cache.values() if row["feasible"]
    ])
    return CompositeOptimizationResult(
        evaluated=evaluated,
        table=pd.DataFrame(table_rows),
        curves=pd.DataFrame(curves),
        history=pd.DataFrame(history),
        report={
            "method": "Seeded genetic algorithm over matrix blend, filler identities, and 0.1 vol% loading genes",
            "objective": (
                "Maximize conservative geometry-normalized capacitance sensitivity: "
                "robust epsilon_r times robust interval-average delta_C/C0 sensitivity"
            ),
            "targets": asdict(targets),
            "selected_material_ids": sorted(selected),
            "settings": asdict(settings),
            "unique_formulations_evaluated": len(cache),
            "feasible_formulations_evaluated": sum(row["feasible"] for row in cache.values()),
            "result_count": len(table_rows),
            "robustness_policy": {
                "screening_ready": "E +/-10%; epsilon_r -10%; Poisson ratio -0.02",
                "provisional": "E +/-25%; epsilon_r -25%; Poisson ratio -0.05",
                "constraint": "Worst-case maximum strain using reduced moduli must not exceed 10%",
            },
            "matrix_blend_model": (
                "Only used when more than one matrix is active. Miscible-blend assumption in 10 vol% "
                "steps of the matrix phase: logarithmic mixing of E and epsilon_r, linear Poisson "
                "ratio, Voigt/Reuss modulus bounds for the stress test."
            ),
            "loading_regime": (
                "Concentrated, 5 to 20 vol% total filler: Bruggeman effective-medium permittivity, "
                "Lewis-Nielsen Young's modulus, self-consistent effective-medium Poisson ratio"
                if settings.loading_regime == "concentrated" else
                "Dilute, below 5 vol% total filler: Maxwell-Garnett permittivity, Mori-Tanaka elasticity"
            ),
            "calculation_mode": "GA evaluated on submission; curves calculated for returned optima",
            "cost_model": False,
        },
    )


def predict_formulation(
    database: MaterialDatabase,
    matrix_percent: dict[str, float],
    filler_vol_percent: dict[str, float],
    targets: Targets,
    regime: str = "dilute",
) -> dict:
    """Nominal model prediction for one formulation, including an unfilled matrix (no fillers)."""
    lookup = {row["material_id"]: row for row in database.material_candidates}
    try:
        matrices = [lookup[key] for key in matrix_percent]
        fillers = [lookup[key] for key in filler_vol_percent]
    except KeyError as exc:
        raise ValueError(f"Unknown material {exc.args[0]}.") from exc
    weights = np.asarray(list(matrix_percent.values()), dtype=float) / 100
    fractions = np.asarray(list(filler_vol_percent.values()), dtype=float) / 100
    matrix = _blend_matrix(matrices, weights)
    filler_e = [_number(row, "youngs_modulus_pa") for row in fillers]
    filler_nu = [_number(row, "poisson_ratio") for row in fillers]
    filler_eps = [_number(row, "relative_permittivity") for row in fillers]
    eps = _effective_permittivity(regime, matrix["eps"], filler_eps, fractions)
    modulus, poisson = _effective_elasticity(regime, matrix["e"], matrix["nu"], filler_e, filler_nu, fractions)
    strain = targets.pressure_max_kpa * 1000 / modulus
    result = {
        "relative_permittivity": eps,
        "youngs_modulus_kpa": modulus / 1000,
        "poisson_ratio": poisson,
        "strain_at_pmax_percent": 100 * strain,
        "sensitivity_per_kpa": np.nan,
        "response_at_pmax": np.nan,
    }
    if strain <= 0.10 + 1e-12:
        response = planar_capacitance_response(
            np.asarray([targets.pressure_min_kpa, targets.pressure_max_kpa]) * 1000, modulus, poisson
        )
        result["sensitivity_per_kpa"] = float(
            (response[1] - response[0]) / (targets.pressure_max_kpa - targets.pressure_min_kpa)
        )
        result["response_at_pmax"] = float(response[1])
    return result
