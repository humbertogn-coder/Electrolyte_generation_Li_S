"""Adaptador PACE-S -> kmc3d.

Traduce un registro válido contra `kmc_interface.schema.json` a los archivos de
entrada del kMC 3D, que no están estandarizados. Se escribe cuando el repo de
kmc3d esté disponible; hasta entonces solo valida y expone los tres campos de
`kinetics` que el kMC necesita.

Fijado a una versión concreta de kmc3d: si cambia, cambia KMC3D_VERSION y se
revisa el mapeo completo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pace_s.contracts import validate

KMC3D_VERSION = "TBD"  # fijar al tag/commit de kmc3d cuando el repo esté disponible

KMC_REQUIRED_KINETICS = ("desolvation_barrier_eV", "reduction_onset_V", "surface_diffusion_barrier_eV")


def kinetics_for_kmc(record: dict[str, Any]) -> dict[str, float]:
    """Los tres parámetros cinéticos que el kMC toma como entrada, tras validar el contrato."""
    validate("kmc_interface", record)
    return {k: float(record["kinetics"][k]) for k in KMC_REQUIRED_KINETICS}


def write_kmc_inputs(record: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Escribe los archivos de entrada del kMC. Pendiente hasta tener el repo de kmc3d."""
    validate("kmc_interface", record)
    raise NotImplementedError(
        f"Mapeo a archivos de kmc3d pendiente (KMC3D_VERSION={KMC3D_VERSION}). "
        "Congelar primero el formato de entrada del kMC."
    )
