"""PACE-S -> kmc3d adapter.

Translates a record valid against `kmc_interface.schema.json` into the input
files of the 3D kMC, which are not standardized. It is written once the kmc3d
repository is available; until then it only validates and exposes the three
`kinetics` fields the kMC needs.

Pinned to a specific kmc3d version: if it changes, KMC3D_VERSION changes and
the full mapping is reviewed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pace_s.contracts import validate

KMC3D_VERSION = "TBD"  # pin to the kmc3d tag/commit once the repository is available

KMC_REQUIRED_KINETICS = ("desolvation_barrier_eV", "reduction_onset_V", "surface_diffusion_barrier_eV")


def kinetics_for_kmc(record: dict[str, Any]) -> dict[str, float]:
    """The three kinetic parameters the kMC takes as input, after validating the contract."""
    validate("kmc_interface", record)
    return {k: float(record["kinetics"][k]) for k in KMC_REQUIRED_KINETICS}


def write_kmc_inputs(record: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Write the kMC input files. Pending until the kmc3d repository is available."""
    validate("kmc_interface", record)
    raise NotImplementedError(
        f"Mapping to kmc3d files pending (KMC3D_VERSION={KMC3D_VERSION}). "
        "Freeze the kMC input format first."
    )
