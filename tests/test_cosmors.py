"""COSMO-RS oracle: structure preparation, mole fractions, vendored pipeline,
and (if opencosmorspy is installed) ln gamma_inf + the gate evaluation on
mock outputs. `tests/data/cosmors/{water,ethanol}.orcacosmo` come from the
openCOSMO-RS_py test suite (LGPL-3.0)."""

import ast
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pace_s.oracles import cosmors

DATA = Path(__file__).parent / "data" / "cosmors"


def _has_opencosmorspy() -> bool:
    try:
        import opencosmorspy  # noqa: F401
        return True
    except ImportError:
        return False


def test_prepare_gate_set(tmp_path):
    df = cosmors.prepare("gate")
    assert df["source"].value_counts().to_dict() == {"joule2021": 15, "solute": 6}
    assert df["inchikey"].is_unique and df["name"].is_unique
    assert (df["charge"] == 0).all()
    inp, csv = cosmors.write_structures(df, tmp_path)
    lines = inp.read_text().splitlines()
    assert len(lines) == len(df)
    name, smiles, xyz, charge, opt = lines[0].split("\t")
    assert name.startswith("Li2S8_") and xyz == "" and charge == "0" and opt == "True"
    assert pd.read_csv(csv).shape[0] == len(df)


def test_prepare_known_set_is_superset():
    gate = cosmors.prepare("gate")
    known = cosmors.prepare("known")
    assert set(gate["inchikey"]) <= set(known["inchikey"])
    assert len(known) > 60
    assert known["name"].str.len().max() <= 40


def test_solutes_are_closed_shell():
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    for smi in cosmors.SOLUTES.values():
        mol = Chem.MolFromSmiles(smi)
        assert Descriptors.NumRadicalElectrons(mol) == 0
        assert sum(a.GetAtomicNum() for a in mol.GetAtoms()) % 2 == 0


def test_mole_fractions():
    x = cosmors.mole_fractions("DOL/DME (1:1 v/v)", ["C1COCO1", "COCCOC"])
    assert abs(sum(x) - 1) < 1e-12 and 0.58 < x[0] < 0.62      # DOL denser and lighter -> more moles
    assert cosmors.mole_fractions("Li(G4)1-TFSI/HFE (1:4 mol)", ["COCCOCCOCCOCCOC", "FC(F)C(F)(F)COC(F)(F)C(F)F"]) == [0.2, 0.8]
    assert cosmors.mole_fractions("THF", ["C1CCOC1"]) == [1.0]
    assert cosmors.mole_fractions("A : B (1:1 v/v)", ["a", "b", "c", "d"]) == [0.25] * 4


def test_vendored_pipeline_parses_and_has_li_radius():
    src = (cosmors.PIPELINE_DIR / "ConformerGenerator.py").read_text()
    ast.parse(src)   # upstream master did not parse when vendored
    radii = dict(line.split("\t") for line in (cosmors.PIPELINE_DIR / "cpcm_radii.inp").read_text().splitlines() if line.strip())
    assert radii["3"] == "2.13" and radii["16"] == "2.16" and radii["9"] == "1.72"


@pytest.mark.skipif(not _has_opencosmorspy(), reason="opencosmorspy not installed")
def test_ln_gamma_inf_ethanol_in_water():
    lng = cosmors.ln_gamma_inf(DATA / "ethanol.orcacosmo", [DATA / "water.orcacosmo"], [1.0])
    assert 1.0 < lng < 3.5     # openCOSMO-RS 24a gives ~2.26; experiment ln(3.7-4.5) ~ 1.3-1.5
    pure = cosmors.ln_gamma_inf(DATA / "ethanol.orcacosmo", [DATA / "ethanol.orcacosmo"], [1.0])
    assert abs(pure) < 1e-6    # pure-component reference state


@pytest.mark.skipif(not _has_opencosmorspy(), reason="opencosmorspy not installed")
def test_evaluate_on_mock_outputs(tmp_path):
    """Every solvent gets the water surface and the solute the ethanol one: the
    evaluation must run over all 27 molecular systems and give one ln gamma each."""
    df = cosmors.prepare("gate")
    for _, r in df.iterrows():
        f = cosmors.orcacosmo_path(tmp_path, r["name"])
        f.parent.mkdir(parents=True)
        shutil.copy(DATA / ("ethanol.orcacosmo" if r["source"] == "solute" else "water.orcacosmo"), f)
    res, stats = cosmors.evaluate(tmp_path, df, "Li2S8")
    assert len(res) == 27
    assert np.allclose(res["ln_gamma_inf_Li2S8"], res["ln_gamma_inf_Li2S8"].iloc[0])  # identical surfaces
    assert set(stats["set"]) == {"all_molecular", "salt<=1.0M"}
