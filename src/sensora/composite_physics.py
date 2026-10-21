"""Small-strain physics used to screen dilute homogeneous composites.

These functions are deliberately independent from the reviewed-device selector.  They
provide a transparent baseline for low-volume-fraction, well-dispersed, approximately
isotropic composites.  They do not model agglomeration, interfacial polarization,
viscoelasticity, piezoelectricity, or pressure-dependent permittivity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class Targets:
    pressure_min_kpa: float = 0.0
    pressure_max_kpa: float = 10.0

    def validate(self) -> None:
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("Pressure limits must be finite and nonnegative.")
        if self.pressure_max_kpa <= self.pressure_min_kpa:
            raise ValueError("Maximum pressure must exceed minimum pressure.")


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _volume_fractions(values: Sequence[float]) -> np.ndarray:
    fractions = np.asarray(values, dtype=float)
    if fractions.ndim != 1 or not np.isfinite(fractions).all():
        raise ValueError("Volume fractions must be a finite one-dimensional sequence.")
    if (fractions < 0).any() or fractions.sum() >= 1:
        raise ValueError("Volume fractions must be nonnegative and sum to less than one.")
    return fractions


def bulk_shear_from_young_poisson(youngs_modulus_pa: float, poisson_ratio: float) -> tuple[float, float]:
    """Convert isotropic Young's modulus and Poisson ratio to bulk and shear moduli."""
    youngs_modulus_pa = _positive_finite(youngs_modulus_pa, "Young's modulus")
    poisson_ratio = float(poisson_ratio)
    if not np.isfinite(poisson_ratio) or not -1 < poisson_ratio < 0.5:
        raise ValueError("Poisson ratio must lie strictly between -1 and 0.5.")
    bulk = youngs_modulus_pa / (3 * (1 - 2 * poisson_ratio))
    shear = youngs_modulus_pa / (2 * (1 + poisson_ratio))
    return bulk, shear


def young_poisson_from_bulk_shear(bulk_modulus_pa: float, shear_modulus_pa: float) -> tuple[float, float]:
    """Convert positive isotropic bulk and shear moduli to Young's modulus and Poisson ratio."""
    bulk_modulus_pa = _positive_finite(bulk_modulus_pa, "Bulk modulus")
    shear_modulus_pa = _positive_finite(shear_modulus_pa, "Shear modulus")
    denominator = 3 * bulk_modulus_pa + shear_modulus_pa
    youngs = 9 * bulk_modulus_pa * shear_modulus_pa / denominator
    poisson = (3 * bulk_modulus_pa - 2 * shear_modulus_pa) / (2 * denominator)
    return youngs, poisson


def maxwell_garnett_permittivity(
    matrix_permittivity: float,
    filler_permittivity: float,
    filler_volume_fraction: float,
) -> float:
    """Maxwell-Garnett permittivity for dilute spherical inclusions."""
    return maxwell_garnett_multiphase(
        matrix_permittivity, [filler_permittivity], [filler_volume_fraction]
    )


def maxwell_garnett_multiphase(
    matrix_permittivity: float,
    filler_permittivities: Sequence[float],
    filler_volume_fractions: Sequence[float],
) -> float:
    """Dilute multi-inclusion Maxwell-Garnett estimate.

    The expression assumes separated spherical inclusions in a continuous matrix.  It
    is appropriate as a baseline at low total loading, not as a claim that particle
    interfaces or clusters are absent in a fabricated composite.
    """
    matrix_permittivity = _positive_finite(matrix_permittivity, "Matrix permittivity")
    fillers = np.asarray(filler_permittivities, dtype=float)
    fractions = _volume_fractions(filler_volume_fractions)
    if fillers.ndim != 1 or len(fillers) != len(fractions) or not np.isfinite(fillers).all():
        raise ValueError("Filler permittivities and fractions must be finite and have equal length.")
    if (fillers <= 0).any():
        raise ValueError("Filler permittivities must be positive.")
    contrast = (fillers - matrix_permittivity) / (fillers + 2 * matrix_permittivity)
    weighted = float(np.dot(fractions, contrast))
    return matrix_permittivity * (1 + 2 * weighted) / (1 - weighted)


def mori_tanaka_elasticity(
    matrix_youngs_modulus_pa: float,
    matrix_poisson_ratio: float,
    filler_youngs_moduli_pa: Sequence[float],
    filler_poisson_ratios: Sequence[float],
    filler_volume_fractions: Sequence[float],
) -> tuple[float, float]:
    """Generalized Mori-Tanaka estimate for isotropic spherical inclusions.

    Returns effective Young's modulus and Poisson ratio.  The implementation uses
    dilute bulk and shear concentration factors normalized by the matrix plus all
    inclusion phases, which avoids an arbitrary ordering for multiple filler types.
    """
    fractions = _volume_fractions(filler_volume_fractions)
    youngs = np.asarray(filler_youngs_moduli_pa, dtype=float)
    poisson = np.asarray(filler_poisson_ratios, dtype=float)
    if youngs.ndim != 1 or poisson.ndim != 1 or not (len(youngs) == len(poisson) == len(fractions)):
        raise ValueError("Filler elastic-property arrays and fractions must have equal length.")
    matrix_bulk, matrix_shear = bulk_shear_from_young_poisson(
        matrix_youngs_modulus_pa, matrix_poisson_ratio
    )
    filler_moduli = [
        bulk_shear_from_young_poisson(value, ratio)
        for value, ratio in zip(youngs, poisson, strict=True)
    ]
    filler_bulk = np.asarray([item[0] for item in filler_moduli])
    filler_shear = np.asarray([item[1] for item in filler_moduli])
    matrix_fraction = 1 - fractions.sum()

    bulk_dilute = (matrix_bulk + 4 * matrix_shear / 3) / (
        filler_bulk + 4 * matrix_shear / 3
    )
    shear_reference = matrix_shear * (9 * matrix_bulk + 8 * matrix_shear) / (
        6 * (matrix_bulk + 2 * matrix_shear)
    )
    shear_dilute = (matrix_shear + shear_reference) / (filler_shear + shear_reference)

    effective_bulk = (
        matrix_fraction * matrix_bulk + np.dot(fractions, filler_bulk * bulk_dilute)
    ) / (matrix_fraction + np.dot(fractions, bulk_dilute))
    effective_shear = (
        matrix_fraction * matrix_shear + np.dot(fractions, filler_shear * shear_dilute)
    ) / (matrix_fraction + np.dot(fractions, shear_dilute))
    return young_poisson_from_bulk_shear(float(effective_bulk), float(effective_shear))


def planar_capacitance_response(
    pressure_pa: float | Sequence[float] | np.ndarray,
    youngs_modulus_pa: float,
    poisson_ratio: float,
    *,
    boundary: str = "free_lateral",
    maximum_linear_strain: float = 0.10,
) -> np.ndarray:
    """Return ``delta_C/C0`` for a constant-permittivity planar capacitor.

    ``fixed_area`` changes capacitance only through thickness. ``free_lateral`` also
    applies the first-order lateral expansion ``(1 + nu * strain)^2``.  Pressure is
    positive in compression.  Values beyond the configured small-strain limit are
    rejected instead of silently extrapolated.
    """
    youngs_modulus_pa = _positive_finite(youngs_modulus_pa, "Young's modulus")
    if boundary not in {"fixed_area", "free_lateral"}:
        raise ValueError("Boundary must be 'fixed_area' or 'free_lateral'.")
    if not 0 < maximum_linear_strain < 1:
        raise ValueError("Maximum linear strain must lie between zero and one.")
    pressure = np.asarray(pressure_pa, dtype=float)
    if not np.isfinite(pressure).all() or (pressure < 0).any():
        raise ValueError("Pressure must be finite and nonnegative.")
    strain = pressure / youngs_modulus_pa
    if (strain > maximum_linear_strain + 1e-12).any():
        raise ValueError(
            f"Pressure exceeds the {maximum_linear_strain:.0%} small-strain limit."
        )
    thickness_factor = 1 / (1 - strain)
    area_factor = 1.0 if boundary == "fixed_area" else (1 + float(poisson_ratio) * strain) ** 2
    return area_factor * thickness_factor - 1
