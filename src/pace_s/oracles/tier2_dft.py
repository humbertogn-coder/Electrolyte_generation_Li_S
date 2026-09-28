"""Tier 2: DFT with ORCA (default) on Li+-solvent complexes, solvent-Li2S8
adducts, reduction potentials and desolvation barriers.

Default level of theory: wB97X-D / def2-TZVPD / CPCM. Never PBE for
potentials or barriers (see oracles/__init__.py).
"""

from __future__ import annotations


def run(candidate_id: str, cfg: dict) -> dict:
    raise NotImplementedError("Tier 2 is implemented in weeks 6-7.")
