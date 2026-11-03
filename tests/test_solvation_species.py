"""Tier 0.5 species: charges, multiplicity, formulas, donor atoms, protocols."""

import pytest

from pace_s.solvation.species import (
    PROTOCOLS,
    Composition,
    Species,
    SpeciesError,
    build_composition,
    cluster_record,
    fsi_anion,
    li_cation,
    make_id,
    n_ligands_to_saturate,
)

DME = Species("DME", "COCCOC", "solvent", "PSM-000001")
THP = Species("THP", "C1CCOCC1", "solvent")
G3 = Species("G3", "COCCOCCOCCOC", "solvent")
TTE = Species("TTE", "FC(F)C(F)(F)OCC(F)(F)C(F)F", "diluent")
DMSO = Species("DMSO", "CS(C)=O", "solvent")
DMA = Species("DMA", "CC(=O)N(C)C", "solvent")
TEA = Species("TEA", "CCN(CC)CC", "solvent")
DES = Species("DES", "CCSCC", "solvent")
SULFOLANE = Species("SL", "O=S1(=O)CCCC1", "solvent")


def test_reference_ions():
    li, fsi = li_cation(), fsi_anion()
    assert (li.charge, li.multiplicity, dict(li.formula)) == (1, 1, {"Li": 1})
    assert fsi.charge == -1 and fsi.multiplicity == 1
    assert dict(fsi.formula) == {"F": 2, "N": 1, "O": 4, "S": 2}
    assert li.n_electrons == 2


def test_closed_and_open_shell():
    assert DME.multiplicity == 1 and DME.n_radical_electrons == 0
    methyl = Species("Me", "[CH3]", "solvent")
    assert methyl.multiplicity == 2 and methyl.n_radical_electrons == 1


def test_invalid_input():
    with pytest.raises(SpeciesError):
        Species("bad", "C1CC", "solvent")
    with pytest.raises(SpeciesError):
        Species("X", "COC", "cosolvent")
    with pytest.raises(SpeciesError):
        Species("D M E", "COCCOC", "solvent")


@pytest.mark.parametrize(
    "sp, n",
    [(DME, 2), (THP, 1), (G3, 4), (TTE, 1), (DMSO, 1), (DMA, 1), (TEA, 1), (DES, 1), (SULFOLANE, 2)],
)
def test_donor_atoms(sp, n):
    """Amide N (DMA), sulfoxide/sulfone S are not donors; amine N and thioether S are."""
    assert sp.n_donor_atoms == n


@pytest.mark.parametrize("sp, n", [(DME, 2), (THP, 4), (G3, 1), (TTE, 4), (SULFOLANE, 2)])
def test_ligands_to_saturate(sp, n):
    assert n_ligands_to_saturate(sp) == n


def test_no_donor_atoms_cannot_saturate():
    with pytest.raises(SpeciesError):
        n_ligands_to_saturate(Species("hexane", "CCCCCC", "diluent"))


@pytest.mark.parametrize(
    "protocol, charge, n_solvent, formula",
    [
        ("isolated_solvent", 0, 1, "C4H10O2"),
        ("isolated_li", 1, 0, "Li"),
        ("isolated_fsi", -1, 0, "F2NO4S2"),
        ("li_s1", 1, 1, "C4H10LiO2"),
        ("li_s2", 1, 2, "C8H20LiO4"),
        ("li_fsi", 0, 0, "F2LiNO4S2"),
        ("li_fsi_s1", 0, 1, "C4H10F2LiNO6S2"),
        ("li_sat", 1, 2, "C8H20LiO4"),
        ("li_fsi_sat", 0, 1, "C4H10F2LiNO6S2"),
    ],
)
def test_protocols_dme(protocol, charge, n_solvent, formula):
    comp = build_composition(protocol, DME)
    assert comp.net_charge == charge
    assert comp.count("solvent") == n_solvent
    assert comp.formula_string == formula
    assert comp.multiplicity == 1


def test_saturated_protocols_depend_on_denticity():
    assert build_composition("li_sat", THP).count("solvent") == 4
    assert build_composition("li_fsi_sat", THP).count("solvent") == 3
    assert build_composition("li_sat", G3).count("solvent") == 1
    assert build_composition("li_fsi_sat", G3).count("solvent") == 0
    # every saturated cluster offers Li+ at least 4 donor atoms (from solvent)
    for sp in (DME, THP, G3, TTE):
        assert build_composition("li_sat", sp).n_donor_atoms_offered >= 4


def test_protocol_errors():
    with pytest.raises(SpeciesError):
        build_composition("li_s3", DME)
    with pytest.raises(SpeciesError):
        build_composition("li_s2")
    with pytest.raises(SpeciesError):
        build_composition("li_s1", fsi_anion())


def test_composition_is_canonical():
    a = Composition.of((DME, 1), li_cation(), (DME, 1))
    b = Composition.of(li_cation(), (DME, 2))
    assert a == b and a.label == b.label == "Li+(1)DME(2)"
    with pytest.raises(SpeciesError):
        Composition.of((DME, 0))
    with pytest.raises(SpeciesError):
        Composition.of(DME, Species("DME", "CCOCCOCC", "solvent"))  # same name, different molecule


def test_cluster_record_derives_charge_and_formula():
    comp = build_composition("li_fsi_s1", DME)
    rec = cluster_record(comp, make_id("PSC", 7), "SPV2-01", "li_fsi_s1", generation_seed=1)
    assert rec["cluster_id"] == "PSC-000007"
    assert rec["net_charge"] == 0 and rec["multiplicity"] == 1
    assert rec["formula"] == "C4H10F2LiNO6S2"
    assert {s["name"]: s["molecule_id"] for s in rec["species"]} == {"Li+": None, "FSI-": None, "DME": "PSM-000001"}


def test_make_id():
    assert make_id("PCC", 311) == "PCC-000311"
    with pytest.raises(ValueError):
        make_id("PSM", 1)
    with pytest.raises(ValueError):
        make_id("PSC", 1_000_000)


def test_every_protocol_is_in_the_schema():
    import json

    from pace_s import CONTRACTS_DIR

    schema = json.loads((CONTRACTS_DIR / "cluster_record.schema.json").read_text(encoding="utf-8"))
    assert set(PROTOCOLS) == set(schema["properties"]["protocol"]["enum"])
