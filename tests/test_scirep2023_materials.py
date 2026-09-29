"""The Sci. Rep. 2023 material list (`data/reference/scirep2023_materials.csv`):
counts, resolved SMILES, and agreement with the seeds it shares."""

import pandas as pd
import pytest
from rdkit import Chem

from pace_s import DATA_DIR
from pace_s.generate.filters import canonical
from pace_s.generate.seeds import load_seeds

CSV = DATA_DIR / "reference" / "scirep2023_materials.csv"


@pytest.fixture(scope="module")
def materials() -> pd.DataFrame:
    return pd.read_csv(CSV)


def test_counts(materials):
    counts = materials["role"].value_counts().to_dict()
    assert counts == {"solvent": 69, "anti_solvent": 5, "salt": 2, "additive": 1}


def test_all_smiles_resolved_and_consistent(materials):
    for _, r in materials.iterrows():
        mol = Chem.MolFromSmiles(r["smiles"])
        assert mol is not None, r["name"]
        assert Chem.MolToInchiKey(mol) == r["inchikey"], r["name"]
    assert materials["inchikey"].is_unique


def test_shared_seeds_agree(materials):
    """Six seeds appear in the Sci. Rep. list; the OPSIN SMILES (from the IUPAC
    name) must match our seed SMILES. This independently confirms them."""
    keys = dict(zip(materials["inchikey"], materials["name"]))
    shared = []
    for s in load_seeds():
        key = Chem.MolToInchiKey(Chem.MolFromSmiles(s.smiles))
        if key in keys:
            shared.append(s.name)
    assert set(shared) == {"DME", "DOL", "TTE", "OFE", "TFTFE", "BTFE"}


def test_salts_are_lithium_salts(materials):
    for _, r in materials[materials["role"] == "salt"].iterrows():
        assert "[Li+]" in canonical(r["smiles"]), r["name"]
