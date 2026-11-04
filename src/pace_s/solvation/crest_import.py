"""Bring CREST conformer searches back as poses re-optimized with xtb.

    python -m pace_s.solvation.crest_import data/campaigns/SPV2-DEMO-MERGED/crest_inputs \
        --out data/campaigns/SPV2-DEMO-CREST --keep 5 --threads 4

The input directory is the one written by `merge_runs --crest-export` after
`workflows/slurm/crest_clusters.sbatch` ran on it: every cluster folder has
meta.json, input.xyz and CREST's crest_conformers.xyz (lowest energy first).

The --keep lowest CREST conformers of every cluster are re-optimized with the
same xtb settings and QC as `run_exchange` (GFN2-xTB/ALPB(ether) by default),
so their energies are directly comparable with the other runs: CREST is used
as a pose generator, not as a different level of theory. Conformers whose
covalent bonds differ from input.xyz (a reaction happened during the
metadynamics) are skipped.

--out becomes a run directory (poses.csv, calculations.jsonl, clusters.jsonl,
run.json) to be merged with the earlier runs by `merge_runs`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from pace_s.contracts import errors as contract_errors
from pace_s.descriptors.geometry import Structure
from pace_s.solvation.cluster_builder import Pose
from pace_s.solvation.convergence import write_csv
from pace_s.solvation.ensemble import bonds
from pace_s.solvation.run_exchange import git_commit, protocol_of, workdir_name
from pace_s.solvation.runner import XtbSettings, run_pose
from pace_s.solvation.species import Composition, Species, cluster_record, make_id

log = logging.getLogger("pace_s.solvation")


def read_multi_xyz(path: Path) -> list[Structure]:
    """All frames of a multi-structure xyz file (CREST's crest_conformers.xyz)."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    out, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        block = lines[i + 2: i + 2 + n]
        symbols = [b.split()[0].capitalize() for b in block]
        coords = np.array([[float(x) for x in b.split()[1:4]] for b in block])
        out.append(Structure(symbols, coords))
        i += 2 + n
    return out


def composition_from_meta(meta: dict) -> Composition:
    return Composition(tuple((Species(p["name"], p["smiles"], p["role"]), int(p["count"])) for p in meta["parts"]))


def ligand_slices(comp: Composition) -> tuple[list[tuple[int, int]], list[str]]:
    """Atom ranges of the ligands in the order cluster_builder writes them (Li+ first)."""
    slices, names, start = [], [], comp.count("li_cation")
    for sp, n in comp.parts:
        if sp.role == "li_cation":
            continue
        k = sp.mol.GetNumAtoms()
        for _ in range(n):
            slices.append((start, start + k))
            names.append(sp.name)
            start += k
    return slices, names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crest_dir", help="folder with one subfolder per cluster (meta.json, crest_conformers.xyz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", type=int, default=5, help="lowest CREST conformers re-optimized per cluster")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--alpb", default="ether")
    ap.add_argument("--campaign", default="SPV2-DEMO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = XtbSettings(alpb=None if args.alpb.lower() == "none" else args.alpb, threads=args.threads)
    out = Path(args.out)
    work = Path(args.workdir) if args.workdir else out / "work"
    out.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    rows, calcs, clusters, solvents = [], [], [], {}
    n = 0
    for d in sorted(p for p in Path(args.crest_dir).iterdir() if (p / "meta.json").exists()):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        conf_file = d / "crest_conformers.xyz"
        if not conf_file.exists():
            log.warning("%s: no crest_conformers.xyz (CREST not run or failed); skipped", d.name)
            continue
        if meta.get("alpb") != settings.alpb:
            log.warning("%s: exported at ALPB=%s but re-optimizing at ALPB=%s", d.name, meta.get("alpb"), settings.alpb)
        comp = composition_from_meta(meta)
        if comp.label != meta["label"]:
            raise SystemExit(f"{d}: meta.json label {meta['label']} does not match its parts ({comp.label})")
        for sp, _n in comp.parts:
            if sp.role in ("solvent", "diluent"):
                solvents.setdefault(sp.name, {"name": sp.name, "smiles": sp.smiles, "role": sp.role})
        ref = Structure.from_xyz(d / "input.xyz")
        ref_bonds = bonds(ref)
        slices, names = ligand_slices(comp)
        frames = read_multi_xyz(conf_file)
        log.info("%-30s %d CREST conformers, re-optimizing up to %d", comp.label, len(frames), args.keep)
        k = 0
        for j, st in enumerate(frames):
            if k >= args.keep:
                break
            if st.symbols != ref.symbols:
                raise SystemExit(f"{d}: CREST changed the atom order or composition")
            if bonds(st) != ref_bonds:
                log.info("   CREST conformer %d: covalent bonds changed during the search; skipped", j)
                continue
            pose = Pose(st, k, slices, names, [{"source": "crest", "crest_rank": j} for _ in slices])
            n += 1
            cid, kid = make_id("PSC", n), make_id("PCC", n)
            res = run_pose(comp, pose, work / workdir_name(comp.label) / f"pose_{k:02d}", settings, cid, kid,
                           args.campaign, commit)
            calc, qc = res["calculation"], res["qc"]
            status = ("optimized" if qc.get("accepted") else
                      "flagged_reactive" if calc["success"] and not qc.get("connectivity_preserved", True) else
                      "excluded" if calc["success"] else "failed")
            crec = cluster_record(comp, cid, args.campaign, protocol_of(comp), 0, k,
                                  geometry_id=calc.get("geometry_id"), status=status,
                                  notes=f"CREST conformer {j} of {meta['source_run']} pose {meta['source_pose']}")
            problems = contract_errors("cluster_record", crec)
            if problems:
                raise ValueError(problems)
            clusters.append(crec)
            calcs.append(calc)
            rows.append({"label": comp.label, "pose_index": k, "cluster_id": cid, "calculation_id": kid,
                         "success": calc["success"], "energy_eh": calc["energy_eh"], "error_code": calc["error_code"],
                         "accepted": qc.get("accepted", False),
                         "connectivity_preserved": qc.get("connectivity_preserved"),
                         "dissociated": qc.get("dissociated"), "li_coordination": qc.get("li_coordination"),
                         "li_f_contacts": qc.get("li_f_contacts"), "denticity_initial": f"crest:{j}",
                         "runtime_s": calc.get("runtime_s")})
            log.info("   CREST conformer %d -> pose %d: E=%s accepted=%s CN(Li)=%s (xtb pose before CREST: %s)",
                     j, k, calc["energy_eh"], qc.get("accepted"), qc.get("li_coordination"), meta["source_energy_eh"])
            k += 1
    if not rows:
        raise SystemExit("no CREST results found")
    write_csv(out / "poses.csv", rows)
    for name, recs in (("clusters.jsonl", clusters), ("calculations.jsonl", calcs)):
        with open(out / name, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in recs)
    (out / "run.json").write_text(json.dumps({"campaign": args.campaign, "commit": commit, "source": "crest",
                                              "crest_dir": str(args.crest_dir), "keep": args.keep,
                                              "method": settings.method, "alpb": settings.alpb,
                                              "workdir": str(work), "solvents": list(solvents.values())},
                                             indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} re-optimized CREST poses to {out}; merge them with the earlier runs:")
    print(f"  python -m pace_s.solvation.merge_runs <run dirs> {out} --out <merged dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
