"""Ligand-exchange (and binding) energies of a few solvents with xtb: the first
physical demonstration of Tier 0.5.

    python -m pace_s.solvation.run_exchange --solvents DME DEE TTE \
        --poses 6 --threads 4 --out data/campaigns/SPV2-DEMO

For every solvent S it builds and optimizes (GFN2-xTB/ALPB(ether) by default):

    S, Li+, FSI-, [Li(S)]+, [Li(S)2]+, [Li(FSI)(S)], and the coordination-
    saturated [Li(S)n]+ and [Li(FSI)(S)(n-1)]  (n from the number of donor atoms)

keeps the lowest-energy accepted pose of each (intact connectivity, no
dissociated ligand) and reports

    dE_ex(n=2)   [Li(S)2]+ + FSI- -> [Li(FSI)(S)] + S
    dE_ex(sat)   [Li(S)n]+ + FSI- -> [Li(FSI)(S)(n-1)] + S
    dE_bind(1)   Li+ + S -> [Li(S)]+

Outputs in --out: reactions.csv (one row per solvent and reaction),
poses.csv (every pose with energy and QC), clusters.jsonl and
calculations.jsonl (records of the PACE-S 2.0 contracts). Working files go to
--workdir (default <out>/work); rerunning the same command skips poses that
already succeeded.

These are electronic energy differences of small clusters inside a reference
continuum ("cluster-in-reference-continuum"), not free energies, CIP/AGG
fractions or electrolyte properties.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

from pace_s import DATA_DIR, REPO_ROOT
from pace_s.contracts import errors as contract_errors
from pace_s.solvation.cluster_builder import build_poses
from pace_s.solvation.ensemble import select_minimum
from pace_s.solvation.reactions import (
    InconsistentEnergiesError,
    MissingEnergyError,
    binding_reaction,
    exchange_reaction,
    reaction_energy,
)
from pace_s.solvation.runner import XtbSettings, run_pose
from pace_s.solvation.species import (
    Composition,
    Species,
    build_composition,
    cluster_record,
    make_id,
    n_ligands_to_saturate,
)

log = logging.getLogger("pace_s.solvation")

# solvents not in data/reference/seeds.csv but used in the Li-SPAN literature
EXTRA_SOLVENTS = {
    "DEE": ("CCOCCOCC", "solvent"),        # 1,2-diethoxyethane
    "DFE": ("COCCOCC(F)F", "solvent"),     # 1,1-difluoro-2-(2-methoxyethoxy)ethane (Li-SPAN, Kim 2026)
    "DBE": ("CCCCOCCCC", "solvent"),
    "DMP": ("COCC(C)OC", "solvent"),
    "BTFEE": ("FC(F)C(F)(F)OCCOC(F)(F)C(F)F", "diluent"),
}


def known_solvents() -> dict[str, tuple[str, str]]:
    out = {}
    with open(DATA_DIR / "reference" / "seeds.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            role = {"seed_solvent": "solvent", "seed_diluent": "diluent"}.get(row["role"], "solvent")
            out[row["name"]] = (row["smiles"], role)
    out.update(EXTRA_SOLVENTS)
    return out


def parse_solvent(arg: str, table: dict[str, tuple[str, str]]) -> Species:
    """'DME' (from the table) or 'NAME=SMILES' (role solvent)."""
    if "=" in arg:
        name, smi = arg.split("=", 1)
        return Species(name, smi, "solvent")
    if arg not in table:
        raise SystemExit(f"unknown solvent {arg!r}; use NAME=SMILES or one of: {', '.join(sorted(table))}")
    smi, role = table[arg]
    return Species(arg, smi, role)


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def reactions_for(solvent: Species) -> list[tuple[str, object]]:
    n_sat = n_ligands_to_saturate(solvent)
    out = [("exchange_n2", exchange_reaction(solvent, 2))]
    if n_sat != 2:
        out.append((f"exchange_sat_n{n_sat}", exchange_reaction(solvent, n_sat)))
    else:
        out.append(("exchange_sat_n2", exchange_reaction(solvent, 2)))
    out.append(("binding_n1", binding_reaction(solvent, 1)))
    return out


def protocol_of(comp: Composition) -> str:
    n_li, n_fsi, n_s = comp.count("li_cation"), comp.count("anion"), comp.count("solvent") + comp.count("diluent")
    if n_li == 0:
        return "isolated_fsi" if n_fsi else "isolated_solvent"
    if n_fsi == 0:
        return {0: "isolated_li", 1: "li_s1", 2: "li_s2"}.get(n_s, "li_sat")
    return {0: "li_fsi", 1: "li_fsi_s1"}.get(n_s, "li_fsi_sat")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solvents", nargs="+", default=["DME", "DEE", "TTE"],
                    help="names from seeds.csv / built-in list, or NAME=SMILES")
    ap.add_argument("--poses", type=int, default=6, help="initial geometries per cluster (default 6)")
    ap.add_argument("--conformers", type=int, default=10, help="RDKit conformers per ligand")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--alpb", default="ether", help="ALPB solvent for xtb; 'none' for gas phase")
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--campaign", default="SPV2-DEMO")
    ap.add_argument("--out", default="data/campaigns/SPV2-DEMO")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    table = known_solvents()
    solvents = [parse_solvent(s, table) for s in args.solvents]
    settings = XtbSettings(alpb=None if args.alpb.lower() == "none" else args.alpb, threads=args.threads)
    out = Path(args.out)
    work = Path(args.workdir) if args.workdir else out / "work"
    out.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    # every composition needed by every reaction, computed once
    plan: list[tuple[Species, str, object]] = [(s, name, rx) for s in solvents for name, rx in reactions_for(s)]
    comps: dict[str, Composition] = {}
    for _, _, rx in plan:
        for comp, _nu in rx.terms():
            comps.setdefault(comp.label, comp)
    log.info("%d solvents, %d reactions, %d distinct clusters/species, up to %d poses each",
             len(solvents), len(plan), len(comps), args.poses)

    clusters, calcs, pose_rows, minima = [], [], [], {}
    n_cluster = n_calc = 0
    for label, comp in sorted(comps.items(), key=lambda kv: sum(kv[1].formula.values())):
        poses = build_poses(comp, n_poses=args.poses, seed=args.seed, n_conformers=args.conformers)
        log.info("%-30s %d atoms, %d poses", label, len(poses[0].structure.symbols), len(poses))
        rows = []
        for pose in poses:
            n_cluster += 1
            n_calc += 1
            cid, kid = make_id("PSC", n_cluster), make_id("PCC", n_calc)
            res = run_pose(comp, pose, work / label.replace("+", "p").replace("-", "m") / f"pose_{pose.pose_index:02d}",
                           settings, cid, kid, args.campaign, commit)
            calc, qc = res["calculation"], res["qc"]
            status = ("optimized" if qc.get("accepted") else
                      "flagged_reactive" if calc["success"] and not qc.get("connectivity_preserved", True) else
                      "excluded" if calc["success"] else "failed")
            crec = cluster_record(comp, cid, args.campaign, protocol_of(comp), args.seed, pose.pose_index,
                                  geometry_id=calc.get("geometry_id"), status=status)
            assert not contract_errors("cluster_record", crec), contract_errors("cluster_record", crec)
            clusters.append(crec)
            calcs.append(calc)
            row = {"label": label, "pose_index": pose.pose_index, "cluster_id": cid, "calculation_id": kid,
                   "success": calc["success"], "energy_eh": calc["energy_eh"], "error_code": calc["error_code"],
                   "accepted": qc.get("accepted", False), "connectivity_preserved": qc.get("connectivity_preserved"),
                   "dissociated": qc.get("dissociated"), "li_coordination": qc.get("li_coordination"),
                   "li_f_contacts": qc.get("li_f_contacts"),
                   "denticity_initial": "+".join(str(b["denticity"]) for b in pose.binding),
                   "runtime_s": calc.get("runtime_s")}
            rows.append(row)
            pose_rows.append(row)
            log.info("   pose %d: %s E=%s accepted=%s CN(Li)=%s %.1f s", pose.pose_index,
                     "ok" if calc["success"] else calc["error_code"], calc["energy_eh"], row["accepted"],
                     row["li_coordination"], calc.get("runtime_s") or 0)
        summary = select_minimum(label, rows)
        best = next((r for r in rows if r["pose_index"] == summary.best_pose), None)
        if best is not None:
            rec = next(c for c in calcs if c["calculation_id"] == best["calculation_id"])
            minima[label] = rec | {"energy_range_eV": summary.energy_range_eV, "n_accepted": summary.n_accepted,
                                   "n_poses": summary.n_poses, "n_near_best": summary.n_near_best,
                                   "n_dissociated": summary.n_dissociated,
                                   "li_coordination": summary.best_li_coordination}
        else:
            log.warning("%s: no accepted pose; reactions using it will be reported as missing", label)

    reaction_rows = []
    for solvent, name, rx in plan:
        row = {"solvent": solvent.name, "smiles": solvent.canonical_smiles, "role": solvent.role,
               "n_donor_atoms": solvent.n_donor_atoms, "reaction": name, "equation": str(rx)}
        try:
            e = reaction_energy(rx, minima)
            terms = [minima[c.label] | {"label": c.label} for c, _ in rx.terms()]
            flags = []
            for t in terms:
                if t["n_poses"] > 1 and t["n_near_best"] < 2:
                    flags.append(f"single_pose_minimum:{t['label']}")
                if t["n_dissociated"] and t["n_dissociated"] >= t["n_poses"] / 2:
                    flags.append(f"ligand_detaches:{t['label']}")
            row.update({"delta_e_eV": round(e["delta_e_eV"], 4), "delta_e_kcal_mol": round(e["delta_e_kcal_mol"], 3),
                        "method": e["method"], "solvation": f"{e['solvation_model']}({e['solvation_reference']})",
                        "min_accepted_poses": min(t["n_accepted"] for t in terms),
                        "min_poses_near_best": min(t["n_near_best"] for t in terms),
                        "flags": ";".join(flags),
                        "status": "ok" if not flags else "ok_with_flags"})
        except MissingEnergyError as err:
            row.update({"status": f"missing: {err}"})
        except InconsistentEnergiesError as err:
            row.update({"status": f"inconsistent: {err}"})
        reaction_rows.append(row)

    def write_csv(path: Path, rows: list[dict]) -> None:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv(out / "reactions.csv", reaction_rows)
    write_csv(out / "poses.csv", pose_rows)
    for name, recs in (("clusters.jsonl", clusters), ("calculations.jsonl", calcs)):
        with open(out / name, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")

    def fmt(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) else "-".rjust(int(spec.split(".")[0]))

    print()
    print(f"{'solvent':8s} {'reaction':16s} {'dE (eV)':>9s} {'dE (kcal/mol)':>14s}  status / flags")
    for r in reaction_rows:
        flags = [f.split(":")[0] for f in (r.get("flags") or "").split(";") if f]
        note = ", ".join(f"{flags.count(k)}x {k}" for k in dict.fromkeys(flags)) or r["status"]
        print(f"{r['solvent']:8s} {r['reaction']:16s} {fmt(r.get('delta_e_eV'), '9.3f')} "
              f"{fmt(r.get('delta_e_kcal_mol'), '14.2f')}  {note[:90]}")
    print("\nsingle_pose_minimum: the lowest pose of a cluster was found by only one starting geometry")
    print("  (sampling not converged; rerun with more --poses or send the cluster to CREST on Grace).")
    print("ligand_detaches: in most poses a ligand left Li+; the cluster is not a stable motif in this continuum.")
    print(f"\nwrote {out / 'reactions.csv'}, poses.csv, clusters.jsonl, calculations.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
