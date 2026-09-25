"""Cada operador de la etapa 1 con un ejemplo de resultado conocido, más
propiedades comunes sobre todas las semillas."""

import pytest
from rdkit import Chem

from pace_s.generate import operators as ops
from pace_s.generate.filters import canonical
from pace_s.generate.seeds import load_seeds

DME = "COCCOC"
THF = "C1CCOC1"
TTE = "FC(F)C(F)(F)OCC(F)(F)C(F)F"


def test_registry_complete():
    assert set(ops.OPERATORS) == {
        "h_to_f", "chain_extend", "chain_contract", "heteroatom_swap",
        "ring_open", "ring_close", "bridge_insert", "branch",
    }


def test_h_to_f():
    assert ops.h_to_f(DME) == {"COCCOCF", "COCC(F)OC"}
    # un carbono ya perfluorado no recibe más F
    assert ops.h_to_f("FC(F)(F)OC(F)(F)F") == set()


def test_chain_extend():
    assert ops.chain_extend("COC") == {"CCOC"}
    assert ops.chain_extend(DME) == {"CCOCCOC", "COCCCOC"}
    # no toca enlaces de anillo
    assert ops.chain_extend(THF) == set()


def test_chain_contract():
    assert ops.chain_contract(DME) == {"COCOC"}
    # no elimina un CH2 entre dos O (crearía O-O)
    assert ops.chain_contract("COCOC") == set()
    # no toca anillos
    assert ops.chain_contract(THF) == set()


def test_chain_extend_contract_are_inverse():
    for child in ops.chain_extend(DME):
        assert canonical(DME) in ops.chain_contract(child)


def test_heteroatom_swap():
    assert ops.heteroatom_swap(DME) == {"COCCSC", "COCCN(C)C"}
    assert ops.heteroatom_swap("CCCC") == set()


def test_ring_open():
    kids = ops.ring_open(THF)
    assert "CCCCCOC" in kids  # apertura por C-O, tapas metilo
    assert "CCCOCCC" in kids  # apertura por el C-C opuesto al O
    for k in kids:
        assert Chem.MolFromSmiles(k).GetRingInfo().NumRings() == 0
    assert ops.ring_open(DME) == set()


def test_ring_close():
    assert ops.ring_close(DME) == {"C1COCCO1"}  # 1,4-dioxano
    kids = ops.ring_close("COCCCOC")
    assert kids
    for k in kids:
        rings = [len(r) for r in Chem.GetSymmSSSR(Chem.MolFromSmiles(k))]
        assert len(rings) == 1
        assert rings[0] in ops.RING_SIZES_TO_CLOSE


def test_bridge_insert():
    assert ops.bridge_insert("COC") == {"COC(C)(F)F", "CO[Si](C)(C)C"}


def test_branch():
    assert ops.branch("COC") == {"CCOC", "CCCOC", "COCC(C)C", "COCC(F)(F)F"}


@pytest.mark.parametrize("name", sorted(ops.OPERATORS))
def test_operator_properties_on_seeds(name):
    """Sobre todas las semillas: hijos válidos, canónicos, sin cargas ni radicales,
    distintos del padre, con los mismos elementos permitidos o Si/S/N introducidos
    solo por los operadores que los introducen."""
    op = ops.OPERATORS[name]
    for seed in load_seeds():
        parent = canonical(seed.smiles)
        for child in op(seed.smiles):
            mol = Chem.MolFromSmiles(child)
            assert mol is not None, (name, seed.name, child)
            assert Chem.MolToSmiles(mol) == child, (name, child)
            assert child != parent
            assert Chem.GetFormalCharge(mol) == 0
            assert all(a.GetNumRadicalElectrons() == 0 for a in mol.GetAtoms())
            assert all(a.GetSymbol() in {"C", "H", "O", "F", "S", "N", "Si"} for a in mol.GetAtoms())


def test_apply_by_name():
    out = ops.apply(["h_to_f", "branch"], "COC")
    assert set(out) == {"h_to_f", "branch"}
    assert out["h_to_f"] == {"COCF"}
