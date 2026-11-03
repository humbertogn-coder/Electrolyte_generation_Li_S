"""Tier 0.5 reactions: conservation of atoms and charge, and honest reaction energies.

Energies below are synthetic numbers chosen to test arithmetic and checks;
they are not results of any calculation.
"""

import pytest

from pace_s.descriptors.xtb import EH_TO_EV
from pace_s.solvation.reactions import (
    InconsistentEnergiesError,
    MissingEnergyError,
    Reaction,
    ReactionBalanceError,
    binding_reaction,
    exchange_reaction,
    reaction_energy,
)
from pace_s.solvation.species import Composition, Species, build_composition, li_cation

DME = Species("DME", "COCCOC", "solvent")
THP = Species("THP", "C1CCOCC1", "solvent")
TTE = Species("TTE", "FC(F)C(F)(F)OCC(F)(F)C(F)F", "diluent")


@pytest.mark.parametrize("solvent", [DME, THP, TTE])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_exchange_is_balanced(solvent, n):
    rx = exchange_reaction(solvent, n)
    assert rx.imbalance() == {}


def test_plan_exchange_reaction_terms():
    rx = exchange_reaction(DME, 2)
    assert str(rx) == "Li+(1)DME(2) + FSI-(1) -> Li+(1)FSI-(1)DME(1) + DME(1)"
    signed = {c.label: nu for c, nu in rx.terms()}
    assert signed == {"Li+(1)DME(2)": -1, "FSI-(1)": -1, "Li+(1)FSI-(1)DME(1)": 1, "DME(1)": 1}


def test_exchange_labels_match_protocol_compositions():
    """The labels of the reaction terms are exactly the clusters built by the protocols,
    so energies computed per protocol can be looked up without renaming."""
    rx = exchange_reaction(DME, 2)
    built = {build_composition(p, DME).label for p in ("li_s2", "isolated_fsi", "li_fsi_s1", "isolated_solvent")}
    assert set(rx.labels()) == built


def test_binding_is_balanced():
    assert binding_reaction(DME, 2).imbalance() == {}


def test_element_imbalance_is_rejected():
    li = Composition.of(li_cation())
    with pytest.raises(ReactionBalanceError, match="C"):
        Reaction("bad", reactants=((li, 1),), products=((Composition.of(li_cation(), DME), 1),))


def test_charge_imbalance_is_rejected():
    li_ion = Composition.of(li_cation())
    li_atom = Composition.of(Species("Li0", "[Li]", "li_cation"))
    with pytest.raises(ReactionBalanceError, match="charge"):
        Reaction("reduction_without_electron", reactants=((li_ion, 1),), products=((li_atom, 1),))


def _rec(label, e, idx, **kw):
    base = {
        "calculation_id": f"PCC-{idx:06d}",
        "success": True,
        "energy_eh": e,
        "method": "GFN2-xTB",
        "solvation_model": "ALPB",
        "solvation_reference": "ether",
        "geometry_type": "xtb_optimized",
        "energy_kind": "electronic_total",
    }
    base.update(kw)
    return base


SYNTHETIC = {  # Hartree, synthetic
    "Li+(1)DME(2)": -10.0,
    "FSI-(1)": -5.0,
    "Li+(1)FSI-(1)DME(1)": -9.1,
    "DME(1)": -6.0,
}


def _energies(**overrides):
    return {lbl: _rec(lbl, e, i) | overrides.get(lbl, {}) for i, (lbl, e) in enumerate(SYNTHETIC.items())}


def test_reaction_energy_arithmetic_and_units():
    out = reaction_energy(exchange_reaction(DME, 2), _energies())
    assert out["delta_e_eh"] == pytest.approx(-9.1 - 6.0 + 10.0 + 5.0)
    assert out["delta_e_eV"] == pytest.approx(out["delta_e_eh"] * EH_TO_EV)
    assert out["method"] == "GFN2-xTB" and out["geometry_type"] == "xtb_optimized"
    assert len(out["calculation_ids"]) == 4


def test_missing_energy_is_an_error_not_zero():
    e = _energies()
    del e["FSI-(1)"]
    with pytest.raises(MissingEnergyError):
        reaction_energy(exchange_reaction(DME, 2), e)


def test_failed_calculation_is_an_error():
    e = _energies(**{"FSI-(1)": {"success": False, "energy_eh": None, "error_code": "SCF_NOT_CONVERGED"}})
    with pytest.raises(MissingEnergyError, match="SCF_NOT_CONVERGED"):
        reaction_energy(exchange_reaction(DME, 2), e)


@pytest.mark.parametrize(
    "key, value",
    [
        ("method", "wB97X-D3/def2-TZVPD"),
        ("solvation_model", "none"),
        ("solvation_reference", "water"),
        ("geometry_type", "dft_single_point_on_xtb_geometry"),
        ("energy_kind", "free_energy_rrho"),
    ],
)
def test_mixed_levels_are_rejected(key, value):
    e = _energies(**{"DME(1)": {key: value}})
    with pytest.raises(InconsistentEnergiesError, match=key):
        reaction_energy(exchange_reaction(DME, 2), e)


def test_wrong_charge_in_record_is_rejected():
    e = _energies(**{"Li+(1)DME(2)": {"charge": 0}})
    with pytest.raises(InconsistentEnergiesError, match="charge"):
        reaction_energy(exchange_reaction(DME, 2), e)
