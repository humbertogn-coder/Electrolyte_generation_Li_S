"""Criterio de éxito de H1: el tier 0 reproduce el orden de DN conocido para
DME, DOL, THP, TTE y F5DEE. Hasta que exista `descriptors.run_tier0`, solo se
comprueba el archivo de referencia; el test de orden queda marcado como skip.
"""

import pandas as pd
import pytest
from rdkit import Chem

from pace_s import DATA_DIR

REF = DATA_DIR / "reference" / "tier0_reference_molecules.csv"


def test_reference_file():
    df = pd.read_csv(REF)
    assert set(df["name"]) == {"DME", "DOL", "THP", "F5DEE", "TTE"}
    for smi in df["smiles"]:
        assert Chem.MolFromSmiles(smi) is not None
    assert set(df["group"]) <= {"coordinating", "weakly_solvating", "diluent"}


@pytest.mark.skip(reason="pendiente: descriptors.run_tier0 (semana 1-2, hito H1)")
def test_tier0_reproduces_donor_order():
    """li_binding_eV: coordinantes < débilmente solvatantes < diluyentes (más negativo = más fuerte)."""
    raise NotImplementedError
