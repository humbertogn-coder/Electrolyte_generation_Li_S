"""Tier 1: box building (Packmol), pre-equilibration (OpenMM/OpenFF), MLIP
molecular dynamics (LAMMPS + plugin) and analysis (MDAnalysis + SolvationAnalysis).

The specific MLIP is decided at milestone H2 (`tiers.tier1.mlip` in the YAML).
"""

from __future__ import annotations


def run(electrolyte_id: str, cfg: dict) -> dict:
    raise NotImplementedError("Tier 1 is implemented after the H2 benchmark (weeks 2-3).")
