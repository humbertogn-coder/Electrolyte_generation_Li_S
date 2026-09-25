"""Regla 4: JSON que valida contra su esquema."""

import copy
import json

import pytest
from jsonschema import ValidationError
from rdkit import Chem

from pace_s import CONTRACTS_DIR
from pace_s.contracts import errors, load_and_validate, validate, validator
from pace_s.export.kmc3d_adapter import KMC_REQUIRED_KINETICS, kinetics_for_kmc

EXAMPLES = CONTRACTS_DIR / "examples"


@pytest.mark.parametrize("name", ["kmc_interface", "candidate_table"])
def test_schemas_are_valid_draft2020(name):
    validator(name)  # check_schema dentro


def test_kmc_example_validates():
    rec = load_and_validate("kmc_interface", EXAMPLES / "PACES-0134.json")
    assert rec["system"] == "Li-S"
    assert rec["provenance"]["tier"] >= 1


def test_candidate_example_validates():
    rec = load_and_validate("candidate_table", EXAMPLES / "PSM-000001.json")
    assert rec["generation"]["stage"] == 0
    # la clave de deduplicación debe ser consistente con RDKit
    mol = Chem.MolFromSmiles(rec["smiles"])
    assert Chem.MolToSmiles(mol) == rec["canonical_smiles"]
    assert Chem.MolToInchiKey(mol) == rec["inchikey"]


def _kmc_example():
    with open(EXAMPLES / "PACES-0134.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("missing", KMC_REQUIRED_KINETICS)
def test_kmc_kinetics_fields_are_required(missing):
    rec = _kmc_example()
    del rec["kinetics"][missing]
    with pytest.raises(ValidationError):
        validate("kmc_interface", rec)


def test_kmc_rejects_tier0():
    rec = _kmc_example()
    rec["provenance"]["tier"] = 0
    assert any("tier" in e for e in errors("kmc_interface", rec))


def test_kmc_rejects_unknown_fields():
    rec = _kmc_example()
    rec["solvation"]["coulombic_efficiency"] = 0.99
    with pytest.raises(ValidationError):
        validate("kmc_interface", rec)


def test_kmc_rejects_bad_id_and_system():
    rec = _kmc_example()
    rec["electrolyte_id"] = "PSM-000001"
    assert errors("kmc_interface", rec)
    rec = _kmc_example()
    rec["system"] = "Li-ion"
    assert errors("kmc_interface", rec)


def test_kmc_fraction_bounds():
    rec = _kmc_example()
    rec["solvation"]["frac_AGG"] = 1.2
    assert errors("kmc_interface", rec)


def test_adapter_exposes_three_kinetic_parameters():
    k = kinetics_for_kmc(_kmc_example())
    assert set(k) == set(KMC_REQUIRED_KINETICS)
    assert all(isinstance(v, float) for v in k.values())


def test_candidate_operator_enum():
    with open(EXAMPLES / "PSM-000001.json", encoding="utf-8") as fh:
        rec = json.load(fh)
    bad = copy.deepcopy(rec)
    bad["generation"]["operator"] = "magic"
    assert errors("candidate_table", bad)
