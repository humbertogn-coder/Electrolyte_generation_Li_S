"""Semillas del espacio de diseño (M0), leídas de `data/reference/seeds.csv`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pace_s import DATA_DIR

SEEDS_CSV = DATA_DIR / "reference" / "seeds.csv"
SALTS_CSV = DATA_DIR / "reference" / "salts.csv"


@dataclass(frozen=True)
class Seed:
    name: str
    role: str  # seed_solvent | seed_diluent
    smiles: str
    full_name: str
    notes: str = ""


def load_seeds(path: str | Path = SEEDS_CSV, include_pending: bool = False) -> list[Seed]:
    """Semillas con SMILES definido. `include_pending=True` incluye filas sin SMILES (p. ej. EMP)."""
    df = pd.read_csv(path, dtype=str).fillna("")
    seeds = [Seed(r["name"], r["role"], r["smiles"], r["full_name"], r["notes"]) for _, r in df.iterrows()]
    if include_pending:
        return seeds
    return [s for s in seeds if s.smiles]


def load_salts(path: str | Path = SALTS_CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")
