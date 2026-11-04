"""Tier 0.5 runner and demo, end to end.

A fake xtb (synthetic energies, geometry unchanged) exercises the whole chain
build -> run -> QC -> records -> reaction energies without xtb installed. One
test runs the real xtb on Li+/DME and is skipped where xtb is not on PATH.
"""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pace_s.contracts import errors
from pace_s.descriptors.xtb import XtbError, find_xtb
from pace_s.solvation import run_exchange
from pace_s.solvation import runner as runner_mod
from pace_s.solvation.cluster_builder import build_poses
from pace_s.solvation.reactions import binding_reaction, reaction_energy
from pace_s.solvation.runner import XtbSettings, run_pose
from pace_s.solvation.species import Species, build_composition

DME = Species("DME", "COCCOC", "solvent")


def fake_xtb(xyz, d, *, charge, uhf, opt, gfn, alpb, threads, timeout_s):
    """Synthetic energy: -1 Eh per atom, -0.05 Eh per unit charge (not physics)."""
    d = Path(d)
    n = int(Path(xyz).read_text().split()[0])
    (d / "xtb.out").write_text("   * xtb version 6.7.1 (fake)\n normal termination of xtb\n", encoding="utf-8")
    (d / "xtbopt.xyz").write_text(Path(xyz).read_text(), encoding="utf-8")
    (d / "wbo").write_text("scratch", encoding="utf-8")
    return SimpleNamespace(energy_eh=-1.0 * n - 0.05 * charge, opt_xyz=d / "xtbopt.xyz")


def failing_xtb(xyz, d, **kw):
    (Path(d) / "xtb.out").write_text("   * xtb version 6.7.1 (fake)\n FAILED TO CONVERGE GEOMETRY OPTIMIZATION\n")
    raise XtbError("xtb did not terminate normally: FAILED TO CONVERGE GEOMETRY OPTIMIZATION")


def test_run_pose_writes_a_valid_record_and_cleans_up(tmp_path):
    comp = build_composition("li_s1", DME)
    pose = build_poses(comp, n_poses=1)[0]
    out = run_pose(comp, pose, tmp_path / "p0", XtbSettings(), "PSC-000001", "PCC-000001", "TEST", runner=fake_xtb)
    rec = out["calculation"]
    assert errors("calculation_record", rec) == []
    assert rec["success"] and rec["energy_eh"] == pytest.approx(-17.05)
    assert rec["program"]["version"] == "6.7.1" and rec["charge"] == 1
    assert out["qc"]["accepted"] and out["qc"]["li_coordination"] >= 1
    assert sorted(f.name for f in (tmp_path / "p0").iterdir()) == ["input.xyz", "result.json", "xtb.out", "xtbopt.xyz"]


def test_run_pose_is_resumable(tmp_path):
    comp = build_composition("li_s1", DME)
    pose = build_poses(comp, n_poses=1)[0]
    run_pose(comp, pose, tmp_path, XtbSettings(), "PSC-000001", "PCC-000001", "TEST", runner=fake_xtb)

    def must_not_run(*a, **k):
        raise AssertionError("a successful pose was recomputed")

    again = run_pose(comp, pose, tmp_path, XtbSettings(), "PSC-000001", "PCC-000001", "TEST", runner=must_not_run)
    assert again["calculation"]["success"]


def test_failure_is_a_record_without_energy(tmp_path):
    comp = build_composition("li_s1", DME)
    pose = build_poses(comp, n_poses=1)[0]
    rec = run_pose(comp, pose, tmp_path, XtbSettings(), "PSC-000002", "PCC-000002", "TEST",
                   runner=failing_xtb)["calculation"]
    assert errors("calculation_record", rec) == []
    assert rec["success"] is False and rec["energy_eh"] is None and rec["error_code"] == "OPT_NOT_CONVERGED"


def test_gas_phase_settings_label_the_record(tmp_path):
    comp = build_composition("isolated_solvent", DME)
    pose = build_poses(comp, n_poses=1)[0]
    rec = run_pose(comp, pose, tmp_path, XtbSettings(alpb=None), "PSC-000003", "PCC-000003", "TEST",
                   runner=fake_xtb)["calculation"]
    assert rec["solvation_model"] == "none" and rec["solvation_reference"] is None
    assert errors("calculation_record", rec) == []


def test_demo_end_to_end_with_fake_xtb(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "run_xtb", fake_xtb)
    rc = run_exchange.main(["--solvents", "DME", "THP", "--poses", "2", "--out", str(tmp_path)])
    assert rc == 0
    with open(tmp_path / "reactions.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["reaction"] for r in rows} >= {"exchange_n2", "binding_n1", "exchange_sat_n4"}
    assert all(r["status"].startswith("ok") for r in rows), rows
    # fake energies: exchange [Li(S)n]+ + FSI- -> [Li(FSI)(S)n-1] + S conserves atoms, so only
    # the charge term (-0.05 Eh per unit charge) survives: 0 - (+1 * -0.05) - (-1 * -0.05) = 0
    assert all(abs(float(r["delta_e_eV"])) < 1e-6 for r in rows if r["reaction"].startswith("exchange"))
    for name, contract in (("clusters.jsonl", "cluster_record"), ("calculations.jsonl", "calculation_record")):
        recs = [json.loads(line) for line in (tmp_path / name).read_text(encoding="utf-8").splitlines()]
        assert recs and all(errors(contract, r) == [] for r in recs)
        ids = [r["cluster_id"] if contract == "cluster_record" else r["calculation_id"] for r in recs]
        assert len(ids) == len(set(ids))


def test_unknown_solvent_message():
    with pytest.raises(SystemExit):
        run_exchange.parse_solvent("XYZ", run_exchange.known_solvents())
    assert run_exchange.parse_solvent("ABC=CCOCC", {}).smiles == "CCOCC"


@pytest.mark.external
@pytest.mark.skipif(find_xtb() is None, reason="xtb is not installed")
def test_real_xtb_li_dme_binding_matches_tier0(tmp_path):
    """Same level as Tier 0 (GFN2-xTB/ALPB ether): Li+ + DME -> [Li(DME)]+ must reproduce
    the regression value of data/reference/tier0_reference_results_gfn2.csv (-2.6655 eV)
    within the conformer/pose noise."""
    rx = binding_reaction(DME, 1)
    energies = {}
    for comp, _ in rx.terms():
        rows = [run_pose(comp, p, tmp_path / comp.label.replace("+", "p") / f"p{p.pose_index}", XtbSettings(threads=2),
                         "PSC-000001", "PCC-000001", "TEST")["calculation"] for p in build_poses(comp, n_poses=3)]
        ok = [r for r in rows if r["success"]]
        energies[comp.label] = min(ok, key=lambda r: r["energy_eh"])
    e = reaction_energy(rx, energies)
    assert e["delta_e_eV"] == pytest.approx(-2.6655, abs=0.08)
