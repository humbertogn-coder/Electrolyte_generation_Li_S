"""Tier 0: parser de xtb (sin binario), geometría (sin binario) e integración
(solo si hay xtb en el PATH o en XTB_BIN)."""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from pace_s.descriptors import geometry as g
from pace_s.descriptors.xtb import XtbError, find_xtb, parse_output

DATA = Path(__file__).parent / "data"
HAS_XTB = find_xtb() is not None


# ----------------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------------


def test_parse_dme_output():
    f = parse_output((DATA / "xtb_dme_opt_alpb.out").read_text())
    assert abs(f["energy_eh"] - (-21.78)) < 0.05
    assert 10.0 < f["gap_eV"] < 12.0
    assert -11.5 < f["homo_eV"] < -10.0
    assert f["lumo_eV"] is not None and f["homo_eV"] < f["lumo_eV"]
    assert f["gsolv_eh"] is not None and f["gsolv_eh"] < 0
    assert 1.0 < f["dipole_D"] < 4.0


def test_parse_rejects_abnormal():
    with pytest.raises(XtbError):
        parse_output("something went wrong\nabnormal termination of xtb\n")


# ----------------------------------------------------------------------------
# geometría
# ----------------------------------------------------------------------------


def test_embed_and_sites():
    mol, cid, n = g.embed_lowest(g.mol_with_hs("COCCOC"), n_conformers=5)
    assert n >= 1
    assert mol.GetNumConformers() == n
    sites = g.coordination_sites(mol)
    assert sites == [1, 4]  # los dos oxígenos de DME (índices del SMILES canónico)
    # con cargas: el más negativo primero
    charges = [0.0] * mol.GetNumAtoms()
    charges[4] = -0.5
    charges[1] = -0.3
    assert g.coordination_sites(mol, charges) == [4, 1]


def test_place_ion_distance_and_direction():
    mol, cid, _ = g.embed_lowest(g.mol_with_hs("COCCOC"), n_conformers=3)
    s = g.Structure.from_mol(mol, cid)
    complex_ = g.place_ion(s, mol, 1, "Li")
    assert complex_.symbols[-1] == "Li"
    assert len(complex_.symbols) == len(s.symbols) + 1
    d = np.linalg.norm(complex_.coords[-1] - s.coords[1])
    assert abs(d - g.ION_DISTANCE_A["O"]) < 1e-6
    # el Li queda más lejos de los vecinos del O que de ese O
    for n in mol.GetAtomWithIdx(1).GetNeighbors():
        assert np.linalg.norm(complex_.coords[-1] - s.coords[n.GetIdx()]) > d


def test_rotation_aligning():
    for a, b in [([1, 0, 0], [0, 1, 0]), ([1, 0, 0], [-1, 0, 0]), ([0, 0, 1], [0, 0, 1]), ([1, 1, 0], [0, 0, 2])]:
        a, b = np.array(a, float), np.array(b, float)
        r = g.rotation_aligning(a, b)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-8)
        assert np.allclose(r @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-8)


def test_place_cluster_no_clash():
    from pace_s.descriptors.run_tier0 import LI2S8_ANCHOR, LI2S8_XYZ

    cluster = g.Structure.from_xyz(LI2S8_XYZ)
    assert cluster.symbols[LI2S8_ANCHOR] == "Li"
    mol, cid, _ = g.embed_lowest(g.mol_with_hs("COCCOC"), n_conformers=3)
    s = g.Structure.from_mol(mol, cid)
    c = g.place_cluster(s, mol, 1, cluster, LI2S8_ANCHOR)
    assert len(c.symbols) == len(s.symbols) + 10
    li = c.coords[len(s.symbols) + LI2S8_ANCHOR]
    assert abs(np.linalg.norm(li - s.coords[1]) - g.ION_DISTANCE_A["O"]) < 0.6
    rest = g.Structure(c.symbols[len(s.symbols):], c.coords[len(s.symbols):])
    assert s.min_distance_to(rest) > 1.5


def test_xyz_roundtrip(tmp_path):
    s = g.Structure(["O", "H", "H"], np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]], float))
    p = s.write_xyz(tmp_path / "w.xyz", "water")
    s2 = g.Structure.from_xyz(p)
    assert s2.symbols == s.symbols
    assert np.allclose(s2.coords, s.coords)


# ----------------------------------------------------------------------------
# referencia (sin xtb: contra los resultados guardados; con xtb: recalculando)
# ----------------------------------------------------------------------------


def _check_reference_order(df):
    """Criterio de éxito de H1 con lo que tier 0 realmente discrimina:
    ESPmin (proxy de DN) ordena coordinantes < débilmente solvatante < diluyente,
    ESPmax marca a los diluyentes, y DME (bidentado) es el que más une Li+."""
    d = df.set_index("name")
    coord = ["DME", "DOL", "THP"]
    assert max(d.loc[coord, "tier0.esp_min_kcal_mol"]) < d.loc["F5DEE", "tier0.esp_min_kcal_mol"]
    assert d.loc["F5DEE", "tier0.esp_min_kcal_mol"] < d.loc["TTE", "tier0.esp_min_kcal_mol"]
    assert d.loc["TTE", "tier0.esp_max_kcal_mol"] > d.loc["F5DEE", "tier0.esp_max_kcal_mol"]
    assert d.loc["F5DEE", "tier0.esp_max_kcal_mol"] > max(d.loc[coord, "tier0.esp_max_kcal_mol"])
    assert d["tier0.li_binding_eV"].idxmin() == "DME"
    assert (d["tier0.li_binding_eV"] < 0).all()
    assert (d["tier0.li2s8_binding_eV"] < 0).all()


def test_reference_results_file():
    import pandas as pd

    from pace_s import DATA_DIR

    df = pd.read_csv(DATA_DIR / "reference" / "tier0_reference_results_gfn2.csv")
    assert {"DME", "DOL", "THP", "F5DEE", "TTE"} <= set(df["name"])
    assert df["tier0.error"].isna().all()
    _check_reference_order(df)


@pytest.mark.skipif(not HAS_XTB, reason="xtb no está instalado")
def test_tier0_dme_with_xtb(tmp_path, cfg_lis):
    from pace_s.descriptors.run_tier0 import Tier0Settings, compute_tier0

    st = Tier0Settings(cfg_lis, threads=2)
    row = compute_tier0("COCCOC", tmp_path / "AL-01" / "tier0" / "DME", st)
    assert row["tier0.error"] is None, row["tier0.error"]
    assert 9.0 < row["tier0.gap_eV"] < 13.0
    assert row["tier0.li_binding_eV"] < -1.0
    assert row["tier0.li2s8_binding_eV"] < 0
    assert row["tier0.esp_min_kcal_mol"] < -30
    assert row["tier0.n_li_sites"] == 2


@pytest.mark.slow
@pytest.mark.skipif(not HAS_XTB, reason="xtb no está instalado")
def test_tier0_reproduces_donor_order_with_xtb(tmp_path, cfg_lis):
    """Hito H1 recalculado (~1 min con 4 hilos). `pytest -m slow` para incluirlo."""
    import pandas as pd

    from pace_s.descriptors.run_tier0 import Tier0Settings, run_block

    st = Tier0Settings(cfg_lis)
    ref = pd.read_csv(Path(__file__).parents[1] / "data" / "reference" / "tier0_reference_molecules.csv")
    items = list(zip(ref["name"], ref["smiles"]))
    df = run_block(items, tmp_path / "AL-01" / "tier0", st).rename(columns={"candidate_id": "name"})
    assert df["tier0.error"].isna().all()
    _check_reference_order(df)
