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


def bruggeman_permittivity(
    matrix_permittivity: float,
    filler_permittivities: Sequence[float],
    filler_volume_fractions: Sequence[float],
) -> float:
    """Bruggeman symmetric effective-medium permittivity for spherical inclusions.

    Solves ``sum_k phi_k (eps_k - eps) / (eps_k + 2 eps) = 0`` over the matrix and every
    filler. The left side falls steadily with ``eps``, so bisection finds the single root.
    """
    matrix_permittivity = _positive_finite(matrix_permittivity, "Matrix permittivity")
    fillers = np.asarray(filler_permittivities, dtype=float)
    fractions = _volume_fractions(filler_volume_fractions)
    if fillers.ndim != 1 or len(fillers) != len(fractions) or not np.isfinite(fillers).all():
        raise ValueError("Filler permittivities and fractions must be finite and have equal length.")
    if (fillers <= 0).any():
        raise ValueError("Filler permittivities must be positive.")
    values = np.concatenate(([matrix_permittivity], fillers))
    shares = np.concatenate(([1 - fractions.sum()], fractions))

    def residual(eps: float) -> float:
        return float(np.sum(shares * (values - eps) / (values + 2 * eps)))

    low, high = float(values.min()), float(values.max())
    if low == high:
        return low
    for _ in range(200):
        middle = 0.5 * (low + high)
        if residual(middle) > 0:
            low = middle
        else:
            high = middle
        if high - low <= 1e-12 * high:
            break
    return 0.5 * (low + high)


def lewis_nielsen_modulus(
    matrix_youngs_modulus_pa: float,
    matrix_poisson_ratio: float,
    filler_youngs_moduli_pa: Sequence[float],
    filler_volume_fractions: Sequence[float],
    *,
    maximum_packing: float = 0.637,
) -> float:
    """Lewis-Nielsen effective modulus of a matrix with spherical filler particles.

    ``E/E_m = (1 + A B phi) / (1 - B psi phi)`` with ``B = (E_f/E_m - 1)/(E_f/E_m + A)`` and
    ``psi = 1 + (1 - phi_max) phi / phi_max**2``. The shape factor ``A`` is Nielsen's value for
    spheres, ``(7 - 5 nu) / (8 - 10 nu)``. The model has one filler, so several fillers are
    treated as one filler with their volume-weighted modulus.
    """
    matrix_modulus = _positive_finite(matrix_youngs_modulus_pa, "Matrix Young's modulus")
    fillers = np.asarray(filler_youngs_moduli_pa, dtype=float)
    fractions = _volume_fractions(filler_volume_fractions)
    if fillers.ndim != 1 or len(fillers) != len(fractions) or not np.isfinite(fillers).all() or (fillers <= 0).any():
        raise ValueError("Filler moduli must be positive and match the volume fractions.")
    nu = float(matrix_poisson_ratio)
    if not np.isfinite(nu) or not -1 < nu < 0.5:
        raise ValueError("Poisson ratio must lie strictly between -1 and 0.5.")
    if not 0 < maximum_packing <= 1:
        raise ValueError("Maximum packing fraction must lie between zero and one.")
    phi = float(fractions.sum())
    if phi == 0:
        return matrix_modulus
    if phi >= maximum_packing:
        raise ValueError("Total filler fraction must be below the maximum packing fraction.")
    filler_modulus = float(np.dot(fractions, fillers) / phi)
    shape = (7 - 5 * nu) / (8 - 10 * nu)
    ratio = filler_modulus / matrix_modulus
    b = (ratio - 1) / (ratio + shape)
    psi = 1 + (1 - maximum_packing) / maximum_packing**2 * phi
    denominator = 1 - b * psi * phi
    if denominator <= 0:
        raise ValueError("The Lewis-Nielsen denominator is not positive for this composition.")
    return float(matrix_modulus * (1 + shape * b * phi) / denominator)


def self_consistent_elasticity(
    matrix_youngs_modulus_pa: float,
    matrix_poisson_ratio: float,
    filler_youngs_moduli_pa: Sequence[float],
    filler_poisson_ratios: Sequence[float],
    filler_volume_fractions: Sequence[float],
) -> tuple[float, float]:
    """Self-consistent (Budiansky-Hill) effective medium for spherical inclusions.

    The matrix and every filler sit in the effective medium itself, which is the elastic
    counterpart of the Bruggeman scheme. Returns the effective Young's modulus and Poisson ratio.
    """
    fractions = _volume_fractions(filler_volume_fractions)
    youngs = np.asarray(filler_youngs_moduli_pa, dtype=float)
    poisson = np.asarray(filler_poisson_ratios, dtype=float)
    if youngs.ndim != 1 or poisson.ndim != 1 or not (len(youngs) == len(poisson) == len(fractions)):
        raise ValueError("Filler elastic-property arrays and fractions must have equal length.")
    moduli = [bulk_shear_from_young_poisson(matrix_youngs_modulus_pa, matrix_poisson_ratio)]
    moduli += [bulk_shear_from_young_poisson(e, nu) for e, nu in zip(youngs, poisson, strict=True)]
    bulk = np.asarray([item[0] for item in moduli])
    shear = np.asarray([item[1] for item in moduli])
    shares = np.concatenate(([1 - fractions.sum()], fractions))
    k, g = float(np.dot(shares, bulk)), float(np.dot(shares, shear))      # Voigt start
    for _ in range(5000):
        k_new = float(np.sum(shares * bulk / (bulk + 4 * g / 3)) / np.sum(shares / (bulk + 4 * g / 3)))
        gamma = g * (9 * k_new + 8 * g) / (6 * (k_new + 2 * g))
        g_new = float(np.sum(shares * shear / (shear + gamma)) / np.sum(shares / (shear + gamma)))
        if abs(k_new - k) <= 1e-12 * k and abs(g_new - g) <= 1e-12 * g:
            k, g = k_new, g_new
            break
        k, g = 0.5 * (k + k_new), 0.5 * (g + g_new)
    else:
        raise ValueError("The self-consistent elasticity iteration did not converge.")
    return young_poisson_from_bulk_shear(k, g)
