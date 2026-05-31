"""Per-species physical parameters used to set up DSMC simulations.

The energy-relaxation and viscosity experiments all need the same handful of
numbers (molecular mass, diameter, Z_rot). Collecting them here means
experiments only have to say `species=Species.H2()` instead of redefining
constants in every script.

Note on Z_rot: the CTC dataset is empirically observed to relax faster than
the Borgnakke-Larssen reference. We expose two values:

    zrot_bl  — value to pass when the collision model is BL
    zrot_mdn — value to pass when the collision model is an MDN/BetaMDN
               (set to zrot_bl / 3.5 to roughly match the CTC relaxation time)
"""

from __future__ import annotations

from dataclasses import dataclass

_AVOGADRO = 6.022e23


@dataclass(frozen=True)
class Species:
    """Physical parameters for a diatomic molecule used by DSMC experiments."""

    name: str
    mass: float  # kg / molecule
    diameter: float  # m
    zrot_bl: float  # Z_rot for the Borgnakke-Larssen collision model
    zrot_mdn: float  # Z_rot for ML collision models

    @classmethod
    def H2(cls) -> "Species":
        zrot_bl = 10
        return cls(
            name="H2",
            mass=2.016e-3 / _AVOGADRO,
            diameter=2.92e-10,
            zrot_bl=zrot_bl,
            zrot_mdn=1,
        )

    @classmethod
    def O2(cls) -> "Species":
        zrot_bl = 4.3
        return cls(
            name="O2",
            mass=32.0e-3 / _AVOGADRO,
            diameter=4.07e-10,
            zrot_bl=zrot_bl,
            zrot_mdn=1,
        )
