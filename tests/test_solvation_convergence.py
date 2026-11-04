"""Sampling convergence, merging of runs and the CREST round trip (fake xtb, fake CREST)."""

import csv
import json
import shutil

import pytest
from test_solvation_runner import fake_xtb

from pace_s.contracts import errors
from pace_s.descriptors.xtb import EH_TO_EV
from pace_s.solvation import crest_import, merge_runs, run_exchange
from pace_s.solvation import runner as runner_mod
from pace_s.solvation.cluster_builder import build_poses
from pace_s.solvation.convergence import assess, normalize_pose_row
from pace_s.solvation.run_exchange import reactions_for
from pace_s.solvation.runner import XtbSettings, run_pose
from pace_s.solvation.species import Species, build_composition

EV = 1 / EH_TO_EV


def rows(energies_ev, run="A", accepted=True):
    return [normalize_pose_row({"pose_index": k, "success": True, "accepted": accepted, "dissociated": False,
                                "energy_eh": -10 + e * EV, "calculation_id": f"PCC-{k:06d}", "li_coordination": 4},
                               run) for k, e in enumerate(energies_ev)]


def test_converged_when_minimum_is_found_early_and_twice():
    c = assess("X", rows([0.0, 0.1, 0.001, 0.2, 0.3, 0.08]), is_cluster=True)
    assert c.converged and c.n_near_best == 2 and c.half_drop_eV == pytest.approx(0.0, abs=1e-9)


def test_minimum_still_dropping_in_the_second_half():
    c = assess("X", rows([0.2, 0.3, 0.25, 0.0, 0.001, 0.4]), is_cluster=True)
    assert not c.converged and "minimum_still_dropping" in c.reasons
    assert c.sampling_drift_eV == pytest.approx(0.2, abs=1e-6)


def test_single_pose_minimum_only_matters_for_clusters():
    r = rows([0.0, 0.1, 0.2, 0.3])
    assert "single_pose_minimum" in assess("X", r, is_cluster=True).reasons
    assert assess("S", r, is_cluster=False).converged


def test_runs_that_disagree_are_not_converged():
    r = rows([0.0, 0.0005, 0.3], "laptop") + rows([0.1, 0.1004, 0.5], "cloud")
    c = assess("X", r, is_cluster=True)
    assert c.run_spread_eV == pytest.approx(0.1, abs=1e-6)
    assert not c.converged and "runs_disagree" in c.reasons


def test_crest_counts_as_converged_and_reports_its_gain():
    r = rows([0.1, 0.3, 0.2], "poses") + rows([0.0, 0.05], "crest")
    c = assess("X", r, is_cluster=True, crest_runs={"crest"})
    assert c.converged and c.crest_checked
    assert c.crest_gain_eV == pytest.approx(0.1, abs=1e-6) and c.best_run == "crest"


def test_no_saturated_exchange_for_diluents():
    tte = Species("TTE", "FC(F)C(F)(F)OCC(F)(F)C(F)F", "diluent")
    dme = Species("DME", "COCCOC", "solvent")
    assert [n for n, _ in reactions_for(tte)] == ["exchange_n2", "binding_n1"]
    assert [n for n, _ in reactions_for(tte, "always")][1].startswith("exchange_sat")
    assert [n for n, _ in reactions_for(dme)] == ["exchange_n2", "exchange_sat_n2", "binding_n1"]


def test_resumed_pose_takes_the_ids_of_the_current_run(tmp_path):
    comp = build_composition("li_s1", Species("DME", "COCCOC", "solvent"))
    pose = build_poses(comp, n_poses=1)[0]
    run_pose(comp, pose, tmp_path, XtbSettings(), "PSC-000001", "PCC-000001", "A", runner=fake_xtb)
    again = run_pose(comp, pose, tmp_path, XtbSettings(), "PSC-000007", "PCC-000009", "B", runner=fake_xtb)
    rec = again["calculation"]
    assert rec["calculation_id"] == "PCC-000009" and rec["geometry_id"] == "PSC-000007-PCC-000009"
    assert errors("calculation_record", rec) == []


def test_merge_crest_export_and_import_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "run_xtb", fake_xtb)
    for seed, name in ((1, "runA"), (2, "runB")):
        assert run_exchange.main(["--solvents", "DME", "--poses", "2", "--isolated-poses", "2",
                                  "--isolated-conformers", "10", "--seed", str(seed), "--out", str(tmp_path / name)]) == 0
    assert (tmp_path / "runA" / "run.json").exists() and (tmp_path / "runA" / "convergence.csv").exists()
    crest_dir = tmp_path / "merged" / "crest_inputs"
    assert merge_runs.main([str(tmp_path / "runA"), str(tmp_path / "runB"), "--out", str(tmp_path / "merged"),
                            "--crest-export", str(crest_dir), "--crest-all"]) == 0
    with open(tmp_path / "merged" / "convergence.csv", encoding="utf-8") as fh:
        conv = {r["label"]: r for r in csv.DictReader(fh)}
    assert conv["Li+(1)DME(2)"]["n_runs"] == "2" and conv["Li+(1)DME(2)"]["n_poses"] == "4"
    with open(tmp_path / "merged" / "reactions.csv", encoding="utf-8") as fh:
        rx = list(csv.DictReader(fh))
    assert {r["reaction"] for r in rx} == {"exchange_n2", "exchange_sat_n2", "binding_n1"}
    assert all(r["sampling_drift_eV"] != "" for r in rx)
    exported = sorted(p.name for p in crest_dir.iterdir())
    assert "Lip(1)DME(2)" in exported and "DME(1)" not in exported   # only Li+ clusters
    meta = json.loads((crest_dir / "Lip(1)DME(2)" / "meta.json").read_text())
    assert meta["net_charge"] == 1 and (crest_dir / "Lip(1)DME(2)" / ".CHRG").read_text().strip() == "1"

    # fake CREST: the conformer file holds the input geometry twice
    for d in crest_dir.iterdir():
        xyz = (d / "input.xyz").read_text()
        (d / "crest_conformers.xyz").write_text(xyz + xyz)
    assert crest_import.main([str(crest_dir), "--out", str(tmp_path / "crest"), "--keep", "2"]) == 0
    calcs = [json.loads(x) for x in (tmp_path / "crest" / "calculations.jsonl").read_text().splitlines()]
    assert calcs and all(errors("calculation_record", c) == [] for c in calcs)
    assert merge_runs.main([str(tmp_path / "runA"), str(tmp_path / "runB"), str(tmp_path / "crest"),
                            "--out", str(tmp_path / "merged2")]) == 0
    with open(tmp_path / "merged2" / "convergence.csv", encoding="utf-8") as fh:
        conv2 = {r["label"]: r for r in csv.DictReader(fh)}
    assert conv2["Li+(1)DME(2)"]["crest_checked"] == "True" and conv2["Li+(1)DME(2)"]["converged"] == "True"


def test_merge_refuses_mixed_levels(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "run_xtb", fake_xtb)
    common = ["--solvents", "DME", "--poses", "1", "--isolated-poses", "1", "--isolated-conformers", "5"]
    run_exchange.main(common + ["--out", str(tmp_path / "ether")])
    run_exchange.main(common + ["--alpb", "none", "--out", str(tmp_path / "gas")])
    with pytest.raises(SystemExit):
        merge_runs.main([str(tmp_path / "ether"), str(tmp_path / "gas"), "--out", str(tmp_path / "m")])
    shutil.rmtree(tmp_path / "m", ignore_errors=True)
