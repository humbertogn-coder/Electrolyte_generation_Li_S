from rdkit import Chem

from pace_s.descriptors.rdkit_descriptors import rdkit_descriptors
from pace_s.generate.filters import fluorination_fraction, hard_filter, longest_perfluoroalkyl_chain
from pace_s.generate.operators import h_to_f


def test_dme_passes(cfg_lis):
    assert hard_filter("COCCOC", cfg_lis["design_space"]).passes


def test_invalid_smiles(cfg_lis):
    assert hard_filter("C(C", cfg_lis["design_space"]).reason == "invalid_smiles"


def test_forbidden_element(cfg_lis):
    r = hard_filter("ClCCOC", cfg_lis["design_space"])
    assert not r.passes and r.reason.startswith("element_not_allowed")


def test_mass_window(cfg_lis):
    ds = cfg_lis["design_space"]
    assert hard_filter("COC", ds).reason.startswith("mol_weight_out_of_range")  # 46 g/mol
    big = "COCCOCCOCCOCCOCCOCCOCCOCCOC"  # hexaglyme, > 350
    assert hard_filter(big, ds).reason.startswith("mol_weight_out_of_range")


def test_forbidden_groups(cfg_lis):
    ds = cfg_lis["design_space"]
    assert hard_filter("N#CCCOCCOC", ds).reason.startswith("forbidden_group")  # nitrile
    assert hard_filter("SCCOCCOC", ds).reason.startswith("forbidden_group")  # thiol


def test_pfas_chain(cfg_lis):
    ds = cfg_lis["design_space"]
    # TTE: only perfluorinated carbons (CF2/CF3) count; the terminal CHF2 do not.
    # Each side of the ether has a single CF2 -> maximum chain 1
    assert longest_perfluoroalkyl_chain(Chem.MolFromSmiles("FC(F)C(F)(F)OCC(F)(F)C(F)F")) == 1
    # OFE: contiguous -(CF2)3- -> 3, at the allowed limit
    assert longest_perfluoroalkyl_chain(Chem.MolFromSmiles("FC(F)C(F)(F)OCC(F)(F)C(F)(F)C(F)(F)C(F)F")) == 3
    # perfluorobutyl ether: (CF2)3CF3 -> 4 > 3 -> fails
    r = hard_filter("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)COCCOC", ds)
    assert r.reason == "perfluoroalkyl_chain_too_long"


def test_fluorination_fraction():
    assert fluorination_fraction(Chem.MolFromSmiles("COCCOC")) == 0.0
    btfe = Chem.MolFromSmiles("FC(F)(F)COCC(F)(F)F")
    assert abs(fluorination_fraction(btfe) - 6 / 10) < 1e-9


def test_rdkit_descriptors_dme():
    d = rdkit_descriptors("COCCOC")
    assert abs(d["mol_weight"] - 90.12) < 0.05
    assert d["ratio_C_O"] == 2.0
    assert d["frac_fluorination"] == 0.0


def test_h_to_f_on_dme():
    children = h_to_f("COCCOC")
    # DME has two non-equivalent positions (terminal CH3, internal CH2)
    assert children == {"COCCOCF", "COCC(F)OC"}
    for c in children:
        assert Chem.MolFromSmiles(c) is not None
