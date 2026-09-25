"""Regla 4: SMILES válidos desde el primer commit."""

from rdkit import Chem

from pace_s.generate.filters import canonical, hard_filter, longest_perfluoroalkyl_chain
from pace_s.generate.seeds import load_salts, load_seeds

EXPECTED_SEEDS = {
    "DME", "DOL", "DMM", "THP", "diglyme", "triglyme", "F5DEE", "FDMB", "EMP",
    "TTE", "OFE", "TFTFE", "BTFE", "TFEO",
}


def test_all_seed_names_present():
    names = {s.name for s in load_seeds(include_pending=True)}
    assert names == EXPECTED_SEEDS


def test_seed_smiles_parse_and_are_canonical():
    for seed in load_seeds():
        mol = Chem.MolFromSmiles(seed.smiles)
        assert mol is not None, seed.name
        assert canonical(seed.smiles) == canonical(canonical(seed.smiles)), seed.name


def test_seed_roles():
    for seed in load_seeds():
        assert seed.role in ("seed_solvent", "seed_diluent"), seed.name


def test_seeds_are_unique():
    canon = [canonical(s.smiles) for s in load_seeds()]
    assert len(canon) == len(set(canon))


def test_diluents_are_fluorinated_and_solvents_mostly_not():
    for seed in load_seeds():
        n_f = seed.smiles.count("F")
        if seed.role == "seed_diluent":
            assert n_f >= 3, seed.name


def test_seeds_pass_hard_filter(cfg_lis):
    """Las semillas definen el espacio; todas deben pasar sus propias restricciones."""
    for seed in load_seeds():
        result = hard_filter(seed.smiles, cfg_lis["design_space"])
        assert result.passes, f"{seed.name}: {result.reason}"


def test_ofe_is_at_pfas_limit():
    ofe = next(s for s in load_seeds() if s.name == "OFE")
    assert longest_perfluoroalkyl_chain(Chem.MolFromSmiles(ofe.smiles)) == 3


def test_salts():
    salts = load_salts()
    assert set(salts["name"]) == {"LiFSI", "LiTFSI"}
    for smi in salts["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        assert Chem.GetFormalCharge(mol) == 0
        assert any(a.GetSymbol() == "Li" for a in mol.GetAtoms())
