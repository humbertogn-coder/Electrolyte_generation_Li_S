"""ORCA single-point export/import on the xtb minima (fake xtb, fake ORCA output)."""

import csv
import json

import pytest
from test_solvation_runner import fake_xtb

from pace_s.contracts import errors
from pace_s.descriptors.xtb import EH_TO_EV
from pace_s.solvation import dft_sp, run_exchange
from pace_s.solvation import runner as runner_mod

ORCA_OK = """
                                 *****************
                                 * O   R   C   A *
                                 *****************
                         Program Version 6.1.1 -  RELEASE  -
...
FINAL SINGLE POINT ENERGY      {e:.12f}
...
                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 0 hours 1 minutes 32 seconds 123 msec
"""


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "run_xtb", fake_xtb)
    assert run_exchange.main(["--solvents", "DME", "--poses", "2", "--isolated-poses", "2",
                              "--isolated-conformers", "10", "--out", str(tmp_path / "run")]) == 0
    out = tmp_path / "dft"
    assert dft_sp.main(["export", str(tmp_path / "run"), "--out", str(out)]) == 0
    return out


def test_export_writes_one_valid_orca_input_per_species(exported):
    manifest = json.loads((exported / "manifest.json").read_text())
    assert set(manifest["labels"]) == {"Li+(1)", "FSI-(1)", "DME(1)", "Li+(1)DME(1)", "Li+(1)DME(2)",
                                       "Li+(1)FSI-(1)DME(1)"}
    inp = (exported / "Lip(1)DME(2)" / "input.inp").read_text().splitlines()
    assert inp[0].startswith("! wB97X-D3 def2-TZVPD def2/J RIJCOSX CPCM")
    assert "* xyz 1 1" in inp and inp[-1] == "*"
    li = (exported / "Lip(1)" / "input.inp").read_text()
    assert "* xyz 1 1\nLi " in li
    meta = json.loads((exported / "Lip(1)FSIm(1)DME(1)" / "meta.json").read_text())
    assert meta["net_charge"] == 0 and meta["xtb"]["method"] == "GFN2-xTB"


def test_import_gives_dft_records_and_reactions_never_mixed_with_xtb(exported):
    # fake ORCA energies: xtb energy - 1 Eh per atom, so every reaction keeps its xtb value
    # except the per-atom term, which cancels because reactions conserve atoms
    for d in exported.iterdir():
        if not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        n = int((d / "geometry.xyz").read_text().split()[0])
        (d / "orca.out").write_text(ORCA_OK.format(e=meta["xtb"]["energy_eh"] - 1.0 * n))
    (exported / "DME(1)" / "orca.out").unlink()          # one calculation never ran
    res = dft_sp.import_results(exported)
    assert len(res["records"]) == 6 and all(errors("calculation_record", r) == [] for r in res["records"])
    ok = [r for r in res["records"] if r["success"]]
    assert len(ok) == 5 and all(r["geometry_type"] == "dft_single_point_on_xtb_geometry" for r in ok)
    assert all(r["program"]["version"] == "6.1.1" and r["runtime_s"] == pytest.approx(92.12, abs=0.01) for r in ok)
    failed = [r for r in res["records"] if not r["success"]]
    assert failed[0]["error_code"] == "NOT_EXECUTED" and failed[0]["energy_eh"] is None
    rows = {r["reaction"]: r for r in res["reactions"]}
    # binding_n1 (Li+ + DME -> Li(DME)+) needs the DME that failed: reported missing at the DFT level
    assert rows["binding_n1"]["delta_e_dft_eV"] is None and "DME(1)" in rows["binding_n1"]["status_dft"]
    # exchange n=2 also needs DME(1); remove the failure and check the values
    meta = json.loads((exported / "DME(1)" / "meta.json").read_text())
    (exported / "DME(1)" / "orca.out").write_text(ORCA_OK.format(e=meta["xtb"]["energy_eh"] - 16.0))
    assert dft_sp.main(["import", str(exported)]) == 0
    with open(exported / "reactions_dft.csv", encoding="utf-8") as fh:
        rows = {r["reaction"]: r for r in csv.DictReader(fh)}
    for r in rows.values():
        assert float(r["dft_minus_xtb_eV"]) == pytest.approx(0.0, abs=1e-6)
    assert float(rows["exchange_n2"]["delta_e_xtb_eV"]) == pytest.approx(0.0, abs=1e-6)  # fake xtb: atoms conserved


def test_orca_failure_codes(tmp_path):
    p = tmp_path / "orca.out"
    assert dft_sp._error_code(dft_sp.parse_orca(p)) == "NOT_EXECUTED"
    p.write_text("Program Version 6.1.1\n ... SCF NOT CONVERGED AFTER 125 CYCLES\n aborting the run\n")
    assert dft_sp._error_code(dft_sp.parse_orca(p)) == "SCF_NOT_CONVERGED"
    assert EH_TO_EV > 27
