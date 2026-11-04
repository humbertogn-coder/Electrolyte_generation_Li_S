"""Merge exchange runs (several machines, seeds or CREST re-optimizations) and
judge the sampling convergence of every cluster.

    python -m pace_s.solvation.merge_runs data/campaigns/SPV2-DEMO-P24 \
        data/campaigns/SPV2-DEMO-P24-CLOUD --out data/campaigns/SPV2-DEMO-MERGED \
        [--crest-export data/campaigns/SPV2-DEMO-MERGED/crest_inputs]

Each RUN_DIR is the --out of `run_exchange` (or of `crest_import`): it holds
poses.csv and calculations.jsonl, plus run.json or reactions.csv (the solvents).
All runs must use the same level of theory (method, continuum, geometry and
energy kind); `reaction_energy` refuses mixtures.

For every cluster the merged set of poses gives the best energy, the per-run
minima and the sampling drift (see `pace_s.solvation.convergence`). Written to
--out: convergence.csv, reactions.csv (with sampling_drift_eV) and merge.json.

--crest-export DIR writes, for every Li+ cluster that is not converged (or for
all of them with --crest-all), DIR/<cluster>/input.xyz (best geometry found so
far), .CHRG, .UHF and meta.json. Copy DIR to Grace and run
workflows/slurm/crest_clusters.sbatch on it; bring the results back with
`python -m pace_s.solvation.crest_import`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

from pace_s.descriptors.geometry import Structure
from pace_s.solvation.convergence import (
    TOL_EV,
    assess,
    normalize_pose_row,
    print_reactions,
    reaction_table,
    write_csv,
)
from pace_s.solvation.run_exchange import is_cluster, reactions_for, workdir_name
from pace_s.solvation.species import Composition, Species

log = logging.getLogger("pace_s.solvation")
LEVEL_KEYS = ("method", "solvation_model", "solvation_reference", "geometry_type", "energy_kind")


class MergeError(ValueError):
    pass


def load_run(run_dir: Path) -> dict[str, Any]:
    """poses (list of dict), calcs (calculation_id -> record), solvents (list of Species)."""
    run_dir = Path(run_dir)
    poses_csv, calcs_jsonl = run_dir / "poses.csv", run_dir / "calculations.jsonl"
    if not poses_csv.exists() or not calcs_jsonl.exists():
        raise MergeError(f"{run_dir} is not a run directory (needs poses.csv and calculations.jsonl)")
    with open(poses_csv, encoding="utf-8") as fh:
        poses = list(csv.DictReader(fh))
    calcs = {}
    for line in calcs_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            calcs[rec["calculation_id"]] = rec
    solvents: list[Species] = []
    source = "poses"
    manifest = run_dir / "run.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        source = m.get("source", "poses")
        solvents = [Species(s["name"], s["smiles"], s["role"]) for s in m.get("solvents", [])]
    elif (run_dir / "reactions.csv").exists():
        with open(run_dir / "reactions.csv", encoding="utf-8") as fh:
            seen = {}
            for r in csv.DictReader(fh):
                seen.setdefault(r["solvent"], (r["smiles"], r["role"]))
        solvents = [Species(n, smi, role) for n, (smi, role) in seen.items()]
    return {"dir": run_dir, "poses": poses, "calcs": calcs, "solvents": solvents, "source": source}


def _run_tags(dirs: list[Path]) -> list[str]:
    tags, used = [], set()
    for d in dirs:
        base, k, tag = Path(d).resolve().name, 1, Path(d).resolve().name
        while tag in used:
            k += 1
            tag = f"{base}#{k}"
        used.add(tag)
        tags.append(tag)
    return tags


def find_geometry(run: dict[str, Any], label: str, pose_index: int, calc: dict[str, Any]) -> Path | None:
    """Optimized geometry of a pose: <run>/work/<cluster>/pose_XX/xtbopt.xyz, else the recorded path."""
    cand = [run["dir"] / "work" / workdir_name(label) / f"pose_{pose_index:02d}" / "xtbopt.xyz"]
    if calc.get("final_geometry_path"):
        cand.append(Path(calc["final_geometry_path"]))
    return next((p for p in cand if p.exists()), None)


def merge(run_dirs: list[Path], saturated: str = "auto", tol: float = TOL_EV) -> dict[str, Any]:
    runs = [load_run(d) for d in run_dirs]
    tags = _run_tags([r["dir"] for r in runs])
    solvents: dict[str, Species] = {}
    for r in runs:
        for s in r["solvents"]:
            if s.name in solvents and solvents[s.name].canonical_smiles != s.canonical_smiles:
                raise MergeError(f"solvent {s.name} has different SMILES in different runs")
            solvents.setdefault(s.name, s)
    if not solvents:
        raise MergeError("no solvent list found in any run (run.json or reactions.csv)")
    plan = [(s, name, rx) for s in solvents.values() for name, rx in reactions_for(s, saturated)]
    comps: dict[str, Composition] = {}
    for _, _, rx in plan:
        for comp, _nu in rx.terms():
            comps.setdefault(comp.label, comp)

    crest_runs = {tag for tag, run in zip(tags, runs) if run["source"] == "crest"}
    rows_by_label: dict[str, list[dict]] = {}
    calc_by_key: dict[tuple[str, str], dict] = {}
    levels = set()
    for tag, run in zip(tags, runs):
        for row in sorted(run["poses"], key=lambda r: int(r["pose_index"])):
            row = normalize_pose_row(row, tag)
            calc = run["calcs"].get(row["calculation_id"])
            if calc is None:
                raise MergeError(f"{run['dir']}: pose row {row['calculation_id']} has no calculation record")
            if calc["success"]:
                levels.add(tuple(calc.get(k) for k in LEVEL_KEYS))
            calc_by_key[(tag, row["calculation_id"])] = calc | {"source_run": tag}
            row["_run_obj"] = run
            rows_by_label.setdefault(row["label"], []).append(row)
    if len(levels) > 1:
        raise MergeError(f"runs mix levels of theory {sorted(levels)}; merge only runs at the same level")

    conv, minima = {}, {}
    for label, comp in comps.items():
        rows = rows_by_label.get(label, [])
        c = assess(label, rows, is_cluster(comp), tol, crest_runs) if rows else None
        if c is None:
            continue
        conv[label] = c
        if c.best_calculation_id is not None:
            minima[label] = calc_by_key[(c.best_run, c.best_calculation_id)]
    reactions = reaction_table(plan, minima, conv)
    return {"tags": tags, "runs": runs, "comps": comps, "rows": rows_by_label, "conv": conv, "minima": minima,
            "reactions": reactions, "solvents": solvents}


def crest_export(merged: dict[str, Any], dest: Path, include_converged: bool = False) -> list[str]:
    """Best geometry of every non-converged Li+ cluster, ready for CREST."""
    dest = Path(dest)
    out = []
    for label, c in merged["conv"].items():
        comp = merged["comps"][label]
        if not is_cluster(comp) or c.best_calculation_id is None or (c.converged and not include_converged):
            continue
        best_row = next(r for r in merged["rows"][label]
                        if r["run"] == c.best_run and r["calculation_id"] == c.best_calculation_id)
        calc = merged["minima"][label]
        geo = find_geometry(best_row["_run_obj"], label, best_row["pose_index"], calc)
        if geo is None:
            log.warning("%s: optimized geometry of the best pose not found; skipped", label)
            continue
        d = dest / workdir_name(label)
        d.mkdir(parents=True, exist_ok=True)
        st = Structure.from_xyz(geo)
        st.write_xyz(d / "input.xyz", f"{label} best xtb pose ({c.best_run} pose {c.best_pose})")
        (d / ".CHRG").write_text(f"{comp.net_charge}\n", encoding="utf-8")
        (d / ".UHF").write_text(f"{comp.multiplicity - 1}\n", encoding="utf-8")
        meta = {"label": label, "net_charge": comp.net_charge, "multiplicity": comp.multiplicity,
                "parts": [{"name": sp.name, "smiles": sp.smiles, "role": sp.role, "count": n} for sp, n in comp.parts],
                "source_run": c.best_run, "source_pose": c.best_pose, "source_energy_eh": c.best_energy_eh,
                "method": calc.get("method"), "alpb": calc.get("solvation_reference"),
                "reasons": c.reasons}
        (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        out.append(label)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (the --out of run_exchange or crest_import)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=TOL_EV)
    ap.add_argument("--saturated", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("--crest-export", default=None, help="write CREST inputs of non-converged clusters here")
    ap.add_argument("--crest-all", action="store_true", help="export every Li+ cluster, converged or not")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        m = merge([Path(r) for r in args.runs], args.saturated, args.tol)
    except MergeError as e:
        raise SystemExit(f"merge_runs: {e}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "convergence.csv", [c.as_row() for c in m["conv"].values()])
    write_csv(out / "reactions.csv", m["reactions"])
    (out / "merge.json").write_text(json.dumps({"runs": [str(r["dir"]) for r in m["runs"]], "tags": m["tags"],
                                                "tol_eV": args.tol, "saturated": args.saturated}, indent=1),
                                    encoding="utf-8")
    print(f"\n{'cluster':32s} {'poses':>5s} {'acc':>4s} {'near':>4s} {'drift':>7s} {'spread':>7s}  converged / why")
    print("  (drift: how much the minimum moved with more poses, between runs, or thanks to CREST; eV)")
    for label, c in sorted(m["conv"].items(), key=lambda kv: kv[0]):
        sp = f"{c.run_spread_eV:7.3f}" if c.run_spread_eV is not None else "      -"
        dr = f"{c.sampling_drift_eV:7.3f}" if c.sampling_drift_eV is not None else "      -"
        print(f"{label:32s} {c.n_poses:5d} {c.n_accepted:4d} {c.n_near_best:4d} {dr} {sp}  "
              f"{('yes (CREST)' if c.crest_checked else 'yes') if c.converged else 'NO: ' + ', '.join(c.reasons)}")
    print_reactions(m["reactions"])
    if args.crest_export:
        dest = Path(args.crest_export)
        done = crest_export(m, dest, args.crest_all)
        stale = sorted(p.name for p in dest.iterdir() if p.is_dir() and p.name not in {workdir_name(x) for x in done})
        if stale:
            log.warning("%s also holds older inputs not exported now: %s", dest, ", ".join(stale))
        print(f"\nCREST inputs for {len(done)} clusters in {dest}")
    print(f"\nwrote {out / 'reactions.csv'}, convergence.csv, merge.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
