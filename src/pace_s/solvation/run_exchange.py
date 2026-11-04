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

The coordination-saturated pair is skipped for diluents (--saturated auto): a
diluent barely solvates Li+, and four of them around Li+ end up detached or
bound through F, which is not a solvation motif. Isolated molecules (S, FSI-)
are cheap and every exchange energy contains them, so they get more conformers
and poses (--isolated-conformers, --isolated-poses).

Also written: convergence.csv (per cluster: best energy, half_drop_eV,
sampling_drift_eV, converged) and run.json (solvents and settings, read by
`merge_runs` to combine runs from several machines or seeds). See
`pace_s.solvation.convergence` for the definitions.

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
from pace_s.solvation.convergence import (
    TOL_EV,
    assess,
    normalize_pose_row,
    print_reactions,
    reaction_table,
    write_csv,
)
from pace_s.solvation.reactions import binding_reaction, exchange_reaction
from pace_s.solvation.runner import XtbSettings, run_pose
from pace_s.solvation.species import (
    Composition,
    Species,
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


def reactions_for(solvent: Species, saturated: str = "auto") -> list[tuple[str, object]]:
    """exchange_n2 and binding_n1 always; the coordination-saturated exchange for
    solvents (saturated='auto'), for every species ('always') or never."""
    out = [("exchange_n2", exchange_reaction(solvent, 2))]
    if saturated == "always" or (saturated == "auto" and solvent.role != "diluent"):
        n_sat = n_ligands_to_saturate(solvent)
        out.append((f"exchange_sat_n{n_sat}", exchange_reaction(solvent, n_sat)))
    out.append(("binding_n1", binding_reaction(solvent, 1)))
    return out


def is_cluster(comp: Composition) -> bool:
    """Li+ with at least one ligand (the poses of these are what CREST can improve)."""
    return comp.count("li_cation") > 0 and comp.n_molecules > 1


def workdir_name(label: str) -> str:
    return label.replace("+", "p").replace("-", "m")


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
    ap.add_argument("--isolated-poses", type=int, default=12,
                    help="minimum number of conformers optimized for isolated S and FSI- (default 12)")
    ap.add_argument("--isolated-conformers", type=int, default=60,
                    help="RDKit embeddings for isolated molecules before pruning (default 60)")
    ap.add_argument("--saturated", choices=["auto", "always", "never"], default="auto",
                    help="coordination-saturated exchange: solvents only (auto), all, or none")
    ap.add_argument("--tol", type=float, default=TOL_EV, help="convergence tolerance in eV (default 0.02)")
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
    plan: list[tuple[Species, str, object]] = [(s, name, rx) for s in solvents
                                                for name, rx in reactions_for(s, args.saturated)]
    comps: dict[str, Composition] = {}
    for _, _, rx in plan:
        for comp, _nu in rx.terms():
            comps.setdefault(comp.label, comp)
    log.info("%d solvents, %d reactions, %d distinct clusters/species, up to %d poses each",
             len(solvents), len(plan), len(comps), args.poses)

    clusters, calcs, pose_rows, minima, conv = [], [], [], {}, {}
    n_cluster = n_calc = 0
    run_tag = out.name
    for label, comp in sorted(comps.items(), key=lambda kv: sum(kv[1].formula.values())):
        if comp.count("li_cation") == 0:
            poses = build_poses(comp, n_poses=max(args.poses, args.isolated_poses), seed=args.seed,
                                n_conformers=max(args.conformers, args.isolated_conformers))
        else:
            poses = build_poses(comp, n_poses=args.poses, seed=args.seed, n_conformers=args.conformers)
        log.info("%-30s %d atoms, %d poses", label, len(poses[0].structure.symbols), len(poses))
        rows = []
        for pose in poses:
            n_cluster += 1
            n_calc += 1
            cid, kid = make_id("PSC", n_cluster), make_id("PCC", n_calc)
            res = run_pose(comp, pose, work / workdir_name(label) / f"pose_{pose.pose_index:02d}",
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
                   "denticity_initial": "+".join(str(b.get("denticity", "")) for b in pose.binding),
                   "runtime_s": calc.get("runtime_s")}
            rows.append(row)
            pose_rows.append(row)
            log.info("   pose %d: %s E=%s accepted=%s CN(Li)=%s %.1f s", pose.pose_index,
                     "ok" if calc["success"] else calc["error_code"], calc["energy_eh"], row["accepted"],
                     row["li_coordination"], calc.get("runtime_s") or 0)
        c = assess(label, [normalize_pose_row(r, run_tag) for r in rows], is_cluster(comp), args.tol)
        conv[label] = c
        if c.best_calculation_id is not None:
            minima[label] = next(x for x in calcs if x["calculation_id"] == c.best_calculation_id)
        else:
            log.warning("%s: no accepted pose; reactions using it will be reported as missing", label)

    reaction_rows = reaction_table(plan, minima, conv)

    write_csv(out / "reactions.csv", reaction_rows)
    write_csv(out / "poses.csv", pose_rows)
    write_csv(out / "convergence.csv", [c.as_row() for c in conv.values()])
    for name, recs in (("clusters.jsonl", clusters), ("calculations.jsonl", calcs)):
        with open(out / name, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in recs)
    manifest = {"campaign": args.campaign, "commit": commit, "seed": args.seed, "poses": args.poses,
                "conformers": args.conformers, "isolated_poses": args.isolated_poses,
                "isolated_conformers": args.isolated_conformers, "saturated": args.saturated,
                "method": settings.method, "alpb": settings.alpb, "workdir": str(work),
                "solvents": [{"name": s.name, "smiles": s.smiles, "role": s.role} for s in solvents]}
    (out / "run.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print_reactions(reaction_rows)
    print(f"\nwrote {out / 'reactions.csv'}, convergence.csv, poses.csv, clusters.jsonl, calculations.jsonl, run.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
