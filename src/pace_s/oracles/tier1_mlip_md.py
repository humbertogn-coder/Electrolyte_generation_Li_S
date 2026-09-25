"""Tier 1: construcción de caja (Packmol), preequilibrado (OpenMM/OpenFF), MD con
MLIP (LAMMPS + plugin) y análisis (MDAnalysis + SolvationAnalysis).

El MLIP concreto se decide en el hito H2 (`tiers.tier1.mlip` en el YAML).
"""

from __future__ import annotations


def run(electrolyte_id: str, cfg: dict) -> dict:
    raise NotImplementedError("Tier 1 se implementa tras el benchmark H2 (semanas 2-3).")
