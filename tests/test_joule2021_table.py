"""Transcription of Table 1 of Joule 2021 (the H3 gate dataset)."""

import pandas as pd
from rdkit import Chem

from pace_s import DATA_DIR

CSV = DATA_DIR / "reference" / "joule2021_table1_solubility.csv"


def test_table_shape():
    df = pd.read_csv(CSV)
    assert df["system_id"].nunique() == 42  # the review says "more than 40 systems"
    assert set(df["category"]) == {"MSE", "SSE", "HSE"}
    assert set(df["species"]) <= {"S8", "Li2S8", "Li2S6", "Li2S4", "Li2S3", "Li2S2", "Li2S"}
    assert (df["solubility_mM_S"] > 0).all()
    assert (df["species"] == "Li2S8").sum() >= 35


def test_solvent_smiles_parse():
    df = pd.read_csv(CSV)
    smiles = df["solvent_smiles"].dropna().unique()
    assert len(smiles) >= 10
    for smi in smiles:
        assert Chem.MolFromSmiles(smi) is not None, smi


def test_known_anchor_values():
    df = pd.read_csv(CSV)

    def value(system, species):
        rows = df[(df["system"] == system) & (df["species"] == species)]["solubility_mM_S"].tolist()
        assert len(rows) == 1, (system, species, rows)
        return rows[0]

    assert value("DMSO", "Li2S8") == 14250
    assert value("Li(G4)1-TFSI/HFE (1:4 mol)", "Li2S8") == 10
    assert value("TTE (70 C)", "Li2S6") == 3.2
    assert value("DME", "S8") == 9.957
    # DOL/DME has three literature values for Li2S8
    assert (df[(df["system"] == "DOL/DME (1:1 v/v)") & (df["species"] == "Li2S8")]).shape[0] == 3
