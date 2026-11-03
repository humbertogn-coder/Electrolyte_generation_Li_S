"""PACE-S 2.0 contracts: cluster, calculation and selection records.

The v1 contracts must stay byte-identical; the new ones must accept complete
records and reject incomplete or contradictory ones (a failed calculation with
an energy, a successful one without provenance). Energies here are synthetic.
"""

import copy
import hashlib
import json

import pytest

from pace_s import CONTRACTS_DIR
from pace_s.contracts import errors, load_and_validate, validator
from pace_s.provenance import stable_hash
from pace_s.solvation.species import Species, build_composition, cluster_record, make_id

EXAMPLES = CONTRACTS_DIR / "examples"
NEW = ["cluster_record", "calculation_record", "selection_record"]

V1_SHA256 = {
    "candidate_table.schema.json": "63c864ddb187682bfc1aec10c2b7c15f2e8bd60ebe5966cbd640dac0d1e7cd47",
    "kmc_interface.schema.json": "8e2e71fa14843f7a66cb6a16ac9cfea8feb9b7777b344973ce1f33536ce0622b",
}


@pytest.mark.parametrize("fname, digest", sorted(V1_SHA256.items()))
def test_v1_contracts_unchanged(fname, digest):
    assert hashlib.sha256((CONTRACTS_DIR / fname).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("name", NEW)
def test_new_schemas_are_valid_draft2020(name):
    validator(name)


@pytest.mark.parametrize(
    "name, fname",
    [
        ("cluster_record", "PSC-000101.json"),
        ("calculation_record", "PCC-000311.json"),
        ("selection_record", "PSL-000001.json"),
    ],
)
def test_examples_validate(name, fname):
    load_and_validate(name, EXAMPLES / fname)


def test_not_executed_example_has_no_energy():
    rec = load_and_validate("calculation_record", EXAMPLES / "PCC-000311.json")
    assert rec["success"] is False and rec["energy_eh"] is None and rec["error_code"] == "NOT_EXECUTED"


def _success_record():
    return {
        "schema_version": "1.0.0",
        "calculation_id": "PCC-000001",
        "cluster_id": "PSC-000001",
        "campaign_id": "TEST",
        "method": "GFN2-xTB",
        "program": {"name": "xtb", "version": "6.7.1"},
        "solvation_model": "ALPB",
        "solvation_reference": "ether",
        "geometry_type": "xtb_optimized",
        "energy_kind": "electronic_total",
        "energy_eh": -12.3456,  # synthetic
        "charge": 1,
        "multiplicity": 1,
        "success": True,
        "error_code": None,
        "geometry_id": "geo-000001-a",
        "input_hash": "0" * 64,
        "command": "xtb in.xyz --opt --alpb ether --chrg 1",
        "log_path": "PCC-000001/xtb.out",
        "log_sha256": "f" * 64,
        "runtime_s": 3.2,
    }


def test_complete_success_record_validates():
    assert errors("calculation_record", _success_record()) == []


@pytest.mark.parametrize("field", ["geometry_id", "input_hash", "command", "log_path", "log_sha256", "runtime_s", "program"])
def test_success_requires_provenance(field):
    rec = _success_record()
    del rec[field]
    assert errors("calculation_record", rec)


def test_success_requires_program_version():
    rec = _success_record()
    del rec["program"]["version"]
    assert errors("calculation_record", rec)


@pytest.mark.parametrize("patch", [{"energy_eh": None}, {"error_code": "OTHER"}])
def test_success_is_contradicted(patch):
    rec = _success_record() | patch
    assert errors("calculation_record", rec)


def test_failure_must_not_carry_an_energy():
    rec = _success_record() | {"success": False, "error_code": "SCF_NOT_CONVERGED"}
    assert errors("calculation_record", rec)  # energy still present
    rec["energy_eh"] = None
    assert errors("calculation_record", rec) == []
    rec["error_code"] = None
    assert errors("calculation_record", rec)  # a failure needs a reason


def test_solvation_reference_consistency():
    rec = _success_record() | {"solvation_model": "none"}
    assert errors("calculation_record", rec)  # reference must be null in gas phase
    rec["solvation_reference"] = None
    assert errors("calculation_record", rec) == []


def test_unknown_fields_and_bad_ids_are_rejected():
    rec = _success_record()
    rec["coulombic_efficiency"] = 0.99
    assert errors("calculation_record", rec)
    rec = _success_record() | {"calculation_id": "PCC-1"}
    assert errors("calculation_record", rec)


def test_cluster_record_from_code_validates():
    dme = Species("DME", "COCCOC", "solvent", "PSM-000001")
    for i, protocol in enumerate(["isolated_solvent", "isolated_li", "isolated_fsi", "li_s1", "li_s2",
                                  "li_fsi", "li_fsi_s1", "li_sat", "li_fsi_sat"]):
        rec = cluster_record(build_composition(protocol, dme), make_id("PSC", i), "TEST", protocol, generation_seed=0)
        assert errors("cluster_record", rec) == [], protocol


def test_cluster_example_is_consistent_with_code():
    """The example file must be exactly what the code produces for [Li(DME)2]+."""
    ex = load_and_validate("cluster_record", EXAMPLES / "PSC-000101.json")
    dme = Species("DME", "COCCOC", "solvent", "PSM-000001")
    rec = cluster_record(build_composition("li_s2", dme), "PSC-000101", ex["campaign_id"], "li_s2",
                         generation_seed=ex["generation_seed"], pose_index=ex.get("pose_index"),
                         geometry_id=ex.get("geometry_id"))
    assert rec == ex


def test_selection_baseline_record():
    rec = load_and_validate("selection_record", EXAMPLES / "PSL-000001.json")
    bad = copy.deepcopy(rec)
    bad["policy"] = "best_guess"
    assert errors("selection_record", bad)


def test_stable_hash():
    a = {"method": "GFN2-xTB", "charge": 1, "xyz": [[0.0, 0.1, -0.0]]}
    b = {"xyz": [[0.0, 0.1 + 1e-13, 0.0]], "charge": 1, "method": "GFN2-xTB"}
    assert stable_hash(a) == stable_hash(b)
    assert stable_hash(a) != stable_hash(a | {"charge": 0})
    assert len(stable_hash(a)) == 64
    with pytest.raises(ValueError):
        stable_hash({"e": float("nan")})
    json.dumps(a)  # inputs stay JSON-serializable
