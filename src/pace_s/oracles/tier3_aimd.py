"""Tier 3: AIMD de interfase Li(100)/electrolito con VASP o CP2K, 20-40 ps.

Presupuesto fijo decidido de antemano (4 sistemas, ~8000 CPU-h). Si aprieta,
3 sistemas y 20 ps. No negociable a mitad.
"""

from __future__ import annotations


def run(electrolyte_id: str, cfg: dict) -> dict:
    raise NotImplementedError("Tier 3 se implementa en semanas 6-7.")
