"""Archivo de moléculas de referencia para el criterio de éxito de H1.
El orden lo comprueba tests/test_tier0.py (contra resultados guardados y,
si hay xtb, recalculando)."""

import pandas as pd
from rdkit import Chem

from pace_s import DATA_DIR

REF = DATA_DIR / "reference" / "tier0_reference_molecules.csv"


def test_reference_file():
    df = pd.read_csv(REF)
    assert set(df["name"]) == {"DME", "DOL", "THP", "F5DEE", "TTE"}
    for smi in df["smiles"]:
        assert Chem.MolFromSmiles(smi) is not None
    assert set(df["group"]) <= {"coordinating", "weakly_solvating", "diluent"}

