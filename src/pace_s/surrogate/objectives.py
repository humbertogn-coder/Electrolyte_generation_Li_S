"""Transformation of raw properties into objectives O1-O5 in the 'higher is better' direction.

The objectives are Pareto, not summed. The YAML weights only enter the qNEHVI
reference weighting and the final ranking.
"""

from __future__ import annotations


def to_objectives(record: dict, cfg: dict) -> dict[str, float]:
    raise NotImplementedError("Implemented in week 4 with the first tier-1 data.")
