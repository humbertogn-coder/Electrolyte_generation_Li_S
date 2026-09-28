"""Synthesizability filter: SAScore (RDKit Contrib) and RAScore (optional).

SAScore (Ertl and Schuffenhauer 2009): 1 easy ... 10 hard. Threshold in the
YAML (`design_space.synthesizability.sascore_max`, 4.5 by default).

RAScore (Thakkar et al. 2021) is not on PyPI; if the `RAscore` package is
installed it is used, otherwise the column stays empty and the threshold is
not applied. Install it from https://github.com/reymond-group/RAscore before H1.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import RDConfig

_SA_DIR = os.path.join(RDConfig.RDContribDir, "SA_Score")
if _SA_DIR not in sys.path:
    sys.path.append(_SA_DIR)

try:
    import sascorer  # type: ignore  # noqa: E402

    _HAS_SA = True
except ImportError:  # pragma: no cover
    _HAS_SA = False


def sascore(smiles: str) -> float | None:
    if not _HAS_SA:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else float(sascorer.calculateScore(mol))


@lru_cache(maxsize=1)
def _rascore_model():
    try:
        from RAscore import RAscore_XGB  # type: ignore

        return RAscore_XGB.RAScorerXGB()
    except Exception:  # noqa: BLE001
        return None


def rascore(smiles: str) -> float | None:
    model = _rascore_model()
    if model is None:
        return None
    try:
        return float(model.predict(smiles))
    except Exception:  # noqa: BLE001
        return None


def passes_synthesizability(sa: float | None, ra: float | None, thresholds: dict) -> bool:
    """True if the available thresholds are met. A missing score does not reject."""
    if sa is not None and sa > thresholds.get("sascore_max", 4.5):
        return False
    if ra is not None and ra < thresholds.get("rascore_min", 0.7):
        return False
    return True
