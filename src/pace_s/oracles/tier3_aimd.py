"""Tier 3: AIMD of the Li(100)/electrolyte interface with VASP or CP2K, 20-40 ps.

Fixed budget decided upfront (4 systems, ~8000 CPU-h). If it gets tight,
3 systems and 20 ps. Not negotiable halfway.
"""

from __future__ import annotations


def run(electrolyte_id: str, cfg: dict) -> dict:
    raise NotImplementedError("Tier 3 is implemented in weeks 6-7.")
