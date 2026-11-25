"""DFT single points (ORCA) on the xtb minima of the clusters, and the same
reaction energies at the DFT level.

    # 1. laptop: one ORCA input per cluster/species, from the merged runs
    python -m pace_s.solvation.dft_sp export RUN_DIR [RUN_DIR ...] --out DFT_DIR
    # 2. Grace: one array task per folder (see workflows/slurm/orca_sp_array.sbatch)
    # 3. laptop: read the ORCA outputs, write records and the reaction table
    python -m pace_s.solvation.dft_sp import DFT_DIR

Level (fixed, ADR D5, already tested on Grace with orca_check.sbatch):
wB97X-D3/def2-TZVPD, RIJCOSX with def2/J, TightSCF, CPCM with the dielectric
constant and refractive index of diethyl ether (4.33, 1.3526), the same
continuum family used by xtb's ALPB(ether).

Geometries are the minima chosen by `merge_runs` (pose sampling + CREST),
optimized with GFN2-xTB/ALPB(ether) and NOT re-optimized with DFT: every DFT
record is labelled geometry_type = dft_single_point_on_xtb_geometry, so
`reaction_energy` never mixes it with xtb energies. The xtb reaction energy of
the same minima is printed next to the DFT one.

Energies are electronic (no ZPE, thermal or entropy terms), like the xtb ones.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pace_s.contracts import errors as contract_errors
from pace_s.descriptors.geometry import Structure
from pace_s.provenance import file_sha256, stable_hash
from pace_s.solvation.convergence import write_csv
from pace_s.solvation.merge_runs import MergeError, find_geometry, merge
from pace_s.solvation.reactions import (
    InconsistentEnergiesError,
    MissingEnergyError,
    reaction_energy,
)
from pace_s.solvation.run_exchange import reactions_for, workdir_name
from pace_s.solvation.species import Species, make_id

log = logging.getLogger("pace_s.solvation")

FUNCTIONAL, BASIS = "wB97X-D3", "def2-TZVPD"
METHOD = f"{FUNCTIONAL}/{BASIS}"
EPSILON, REFRAC = 4.33, 1.3526
SOLVATION_REFERENCE = f"diethyl ether (epsilon={EPSILON}, n={REFRAC})"
MAXCORE_MB = 3000
HEADER = (f"! {FUNCTIONAL} {BASIS} def2/J RIJCOSX CPCM TightSCF\n"
          f"%maxcore {MAXCORE_MB}\n"
          f"%cpcm epsilon {EPSILON} refrac {REFRAC} end\n")
# the sbatch inserts "%pal nprocs N end" after the first line

_RE_ENERGY = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_RE_VERSION = re.compile(r"Program Version\s+([0-9][0-9.]*)")
_RE_RUNTIME = re.compile(r"TOTAL RUN TIME:\s+(\d+) days (\d+) hours (\d+) minutes (\d+) seconds (\d+) msec")


def orca_input(structure: Structure, charge: int, multiplicity: int, title: str) -> str:
    lines = HEADER.rstrip("\n").split("\n")   # the '!' line stays first
    lines.insert(1, f"# {title}")
    lines.append(f"* xyz {charge} {multiplicity}")
    lines += [f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}" for s, (x, y, z) in zip(structure.symbols, structure.coords)]
    lines.append("*")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- export

def export(run_dirs: list[Path], out: Path, saturated: str = "auto") -> list[str]:
    m = merge(run_dirs, saturated)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for label, comp in sorted(m["comps"].items()):
        c = m["conv"].get(label)
        if c is None or c.best_calculation_id is None:
            log.warning("%s: no accepted minimum; not exported", label)
            continue
        calc = m["minima"][label]
        if sum(comp.formula.values()) == 1:
            sym = next(iter(comp.formula))
            st = Structure([sym], np.zeros((1, 3)))
        else:
            row = next(r for r in m["rows"][label] if r["run"] == c.best_run and r["calculation_id"] == c.best_calculation_id)
            geo = find_geometry(row["_run_obj"], label, row["pose_index"], calc)
            if geo is None:
                log.warning("%s: optimized geometry not found; not exported", label)
                continue
            st = Structure.from_xyz(geo)
            st = Structure([s.capitalize() for s in st.symbols], st.coords)
        d = out / workdir_name(label)
        d.mkdir(parents=True, exist_ok=True)
        st.write_xyz(d / "geometry.xyz", f"{label} xtb minimum ({c.best_run}, {c.best_calculation_id})")
        (d / "input.inp").write_text(orca_input(st, comp.net_charge, comp.multiplicity, label), encoding="utf-8")
        xtb = {k: calc.get(k) for k in ("calculation_id", "cluster_id", "geometry_id", "method", "solvation_model",
                                        "solvation_reference", "geometry_type", "energy_kind", "energy_eh",
                                        "charge", "multiplicity", "success")}
        meta = {"label": label, "net_charge": comp.net_charge, "multiplicity": comp.multiplicity,
                "parts": [{"name": sp.name, "smiles": sp.smiles, "role": sp.role, "count": n} for sp, n in comp.parts],
                "source_run": c.best_run, "crest_checked": c.crest_checked, "converged": c.converged,
                "geometry_sha256": file_sha256(d / "geometry.xyz"), "xtb": xtb}
        (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        done.append(label)
    manifest = {"method": METHOD, "header": HEADER, "solvation_reference": SOLVATION_REFERENCE,
                "saturated": saturated, "runs": [str(r["dir"]) for r in m["runs"]],
                "solvents": [{"name": s.name, "smiles": s.smiles, "role": s.role} for s in m["solvents"].values()],
                "labels": done}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return done


# --------------------------------------------------------------------------- import

def parse_orca(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    e = _RE_ENERGY.findall(text)
    v = _RE_VERSION.search(text)
    t = _RE_RUNTIME.search(text)
    runtime = None
    if t:
        dd, hh, mm, ss, ms = (int(x) for x in t.groups())
        runtime = dd * 86400 + hh * 3600 + mm * 60 + ss + ms / 1000
    return {"exists": bool(text), "normal": "ORCA TERMINATED NORMALLY" in text,
            "energy_eh": float(e[-1]) if e else None, "version": v.group(1) if v else None,
            "runtime_s": runtime, "text_tail": text[-2000:]}


def _error_code(p: dict[str, Any]) -> str:
    if not p["exists"]:
        return "NOT_EXECUTED"
    tail = p["text_tail"].upper()
    if "SCF NOT CONVERGED" in tail or "SCF NOT FULLY CONVERGED" in tail:
        return "SCF_NOT_CONVERGED"
    if p["normal"] and p["energy_eh"] is None:
        return "PARSE_ERROR"
    return "PROGRAM_ERROR"


def dft_record(d: Path, meta: dict, calc_id: str, cluster_id: str, campaign: str) -> dict[str, Any]:
    inp = (d / "input.inp").read_text(encoding="utf-8")
    p = parse_orca(d / "orca.out")
    rec: dict[str, Any] = {
        "schema_version": "1.0.0", "calculation_id": calc_id, "cluster_id": cluster_id, "campaign_id": campaign,
        "method": METHOD, "program": {"name": "orca"}, "solvation_model": "CPCM",
        "solvation_reference": SOLVATION_REFERENCE, "geometry_type": "dft_single_point_on_xtb_geometry",
        "energy_kind": "electronic_total", "energy_eh": None, "charge": meta["net_charge"],
        "multiplicity": meta["multiplicity"], "success": False, "error_code": None,
        "input_hash": stable_hash({"input": inp}), "command": "orca run.inp (run.inp = input.inp + %pal nprocs)",
        "input_geometry_path": str(d / "geometry.xyz"),
        "parameters": {"aux_basis": "def2/J", "rijcosx": True, "scf": "TightSCF", "epsilon": EPSILON,
                       "refrac": REFRAC, "maxcore_mb": MAXCORE_MB,
                       "xtb_geometry_from": meta["xtb"].get("calculation_id"),
                       "xtb_source_run": meta.get("source_run")},
    }
    if p["version"]:
        rec["program"]["version"] = p["version"]
    if p["exists"]:
        rec["log_path"] = str(d / "orca.out")
        rec["log_sha256"] = file_sha256(d / "orca.out")
    if p["runtime_s"] is not None:
        rec["runtime_s"] = round(p["runtime_s"], 2)
    if p["normal"] and p["energy_eh"] is not None:
        rec.update({"success": True, "energy_eh": p["energy_eh"],
                    "geometry_id": meta["xtb"].get("geometry_id") or f"sha256:{meta['geometry_sha256'][:16]}"})
        rec.setdefault("program", {}).setdefault("version", "unknown")
        rec.setdefault("runtime_s", 0.0)
    else:
        rec["error_code"] = _error_code(p)
        rec["error_message"] = p["text_tail"][-500:] if p["exists"] else "orca.out not found"
    return rec


def import_results(dft_dir: Path, campaign: str = "SPV2-DEMO") -> dict[str, Any]:
    dft_dir = Path(dft_dir)
    manifest = json.loads((dft_dir / "manifest.json").read_text(encoding="utf-8"))
    solvents = [Species(s["name"], s["smiles"], s["role"]) for s in manifest["solvents"]]
    plan = [(s, name, rx) for s in solvents for name, rx in reactions_for(s, manifest.get("saturated", "auto"))]
    dft, xtb, recs = {}, {}, []
    for k, d in enumerate(sorted(p for p in dft_dir.iterdir() if (p / "meta.json").exists()), start=1):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        rec = dft_record(d, meta, make_id("PCC", k), make_id("PSC", k), campaign)
        problems = contract_errors("calculation_record", rec)
        if problems:
            raise ValueError(f"{d}: DFT record does not satisfy the contract: {problems}")
        recs.append(rec)
        dft[meta["label"]] = rec
        xtb[meta["label"]] = meta["xtb"]
    rows = []
    for solvent, name, rx in plan:
        row: dict[str, Any] = {"solvent": solvent.name, "role": solvent.role, "reaction": name, "equation": str(rx)}
        for tag, energies in (("xtb", xtb), ("dft", dft)):
            try:
                e = reaction_energy(rx, energies)
                row[f"delta_e_{tag}_eV"] = round(e["delta_e_eV"], 4)
            except (MissingEnergyError, InconsistentEnergiesError) as err:
                row[f"delta_e_{tag}_eV"] = None
                row[f"status_{tag}"] = str(err)
        if row.get("delta_e_xtb_eV") is not None and row.get("delta_e_dft_eV") is not None:
            row["dft_minus_xtb_eV"] = round(row["delta_e_dft_eV"] - row["delta_e_xtb_eV"], 4)
        rows.append(row)
    return {"records": recs, "reactions": rows, "manifest": manifest}


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write one ORCA input per cluster/species from merged runs")
    ex.add_argument("runs", nargs="+")
    ex.add_argument("--out", required=True)
    ex.add_argument("--saturated", choices=["auto", "always", "never"], default="auto")
    im = sub.add_parser("import", help="read ORCA outputs; write records and reactions")
    im.add_argument("dft_dir")
    im.add_argument("--campaign", default="SPV2-DEMO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "export":
        try:
            done = export([Path(r) for r in args.runs], Path(args.out), args.saturated)
        except MergeError as e:
            raise SystemExit(f"dft_sp export: {e}")
        print(f"\n{len(done)} ORCA inputs ({METHOD}, CPCM {SOLVATION_REFERENCE}) in {args.out}")
        print("Copy the folder to Grace and submit (see workflows/slurm/orca_sp_array.sbatch):")
        print(f"  sbatch --array=0-{len(done) - 1} workflows/slurm/orca_sp_array.sbatch /scratch/user/$USER/<folder>")
        return 0

    res = import_results(Path(args.dft_dir), args.campaign)
    d = Path(args.dft_dir)
    with open(d / "calculations_dft.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in res["records"])
    write_csv(d / "reactions_dft.csv", res["reactions"])
    failed = [r for r in res["records"] if not r["success"]]
    print(f"\n{len(res['records']) - len(failed)} of {len(res['records'])} ORCA single points succeeded")
    for r in failed:
        print(f"  FAILED {r['input_geometry_path']}: {r['error_code']}")

    def f(v):
        return f"{v:9.3f}" if isinstance(v, (int, float)) else "        -"

    print(f"\n{'solvent':8s} {'reaction':16s} {'xtb (eV)':>9s} {'DFT (eV)':>9s} {'DFT-xtb':>9s}")
    for r in res["reactions"]:
        print(f"{r['solvent']:8s} {r['reaction']:16s} {f(r.get('delta_e_xtb_eV'))} {f(r.get('delta_e_dft_eV'))} "
              f"{f(r.get('dft_minus_xtb_eV'))}")
    print(f"\nDFT: {METHOD}, CPCM {SOLVATION_REFERENCE}, single points on GFN2-xTB/ALPB(ether) minima;")
    print("electronic energies only (no ZPE, thermal or entropy terms).")
    print(f"wrote {d / 'reactions_dft.csv'} and calculations_dft.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
