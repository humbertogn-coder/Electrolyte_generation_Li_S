"""Tier 2: DFT con ORCA (por defecto) sobre complejos Li+-solvente, aductos
solvente-Li2S8, potenciales de reducción y barreras de desolvatación.

Nivel de teoría por defecto: ωB97X-D / def2-TZVPD / CPCM. Nunca PBE para
potenciales o barreras (ver oracles/__init__.py).
"""

from __future__ import annotations


def run(candidate_id: str, cfg: dict) -> dict:
    raise NotImplementedError("Tier 2 se implementa en semanas 6-7.")
