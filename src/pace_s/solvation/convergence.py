"""Sampling convergence of cluster minima, merging of runs, and reaction tables
with a sampling-drift estimate.

A run of `run_exchange` keeps, for every cluster, the lowest-energy accepted
pose. Whether that is the true minimum of the cluster at this level depends on
how many starting geometries were tried. Two observable signals are used here:

- half_drop_eV: E_min(first half of the accepted poses, in sampling order)
  minus E_min(all accepted poses). If doubling the number of poses still lowers
  the minimum, sampling is not converged.
- run_spread_eV: spread of the per-run minima when several independent runs
  (different machines or --seed) are merged.

sampling_drift_eV = max(half_drop_eV, run_spread_eV) is how much the minimum
was *seen* to move. It is a lower bound of the sampling error, not a
statistical error bar: a valley that no pose reached is invisible to it.

A cluster is `converged` when the drift is <= tol (default 0.02 eV) and, for a
Li+ cluster with ligands, at least two accepted poses reached the minimum
(within NEAR_BEST_EV). A single atom (Li+) is converged by construction.

Clusters searched with CREST (poses from a run made by `crest_import`) are the
exception: CREST's metadynamics is the most thorough search available at this
level, so the cluster counts as converged once one CREST pose is accepted, and
the drift is how much CREST lowered the minimum of the pose sampling
(crest_gain_eV). A large gain means the earlier pose sampling was poor.

For a reaction, the drift of each term is added with |stoichiometric
coefficient| (conservative: the errors are not assumed to cancel).
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pace_s.descriptors.xtb import EH_TO_EV
from pace_s.solvation.ensemble import NEAR_BEST_EV
from pace_s.solvation.reactions import (
    InconsistentEnergiesError,
    MissingEnergyError,
    Reaction,
    reaction_energy,
)

TOL_EV = 0.02


@dataclass
class ClusterConvergence:
    label: str
    is_cluster: bool                 # contains Li+ and at least one ligand
    n_runs: int
    n_poses: int
    n_succeeded: int
    n_accepted: int
    n_dissociated: int
    n_near_best: int
    best_energy_eh: float | None
    best_run: str | None
    best_pose: int | None
    best_calculation_id: str | None
    best_li_coordination: int | None
    energy_range_eV: float | None
    half_drop_eV: float | None
    run_spread_eV: float | None
    sampling_drift_eV: float | None
    converged: bool
    reasons: list[str] = field(default_factory=list)
    run_minima_eh: dict[str, float] = field(default_factory=dict)
    crest_checked: bool = False
    crest_gain_eV: float | None = None

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["reasons"] = ";".join(self.reasons)
        row["run_minima_eh"] = json.dumps(self.run_minima_eh)
        for k in ("energy_range_eV", "half_drop_eV", "run_spread_eV", "sampling_drift_eV", "crest_gain_eV"):
            if row[k] is not None:
                row[k] = round(row[k], 4)
        return row


def _flag(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def _int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    return int(float(v))


def normalize_pose_row(row: dict[str, Any], run: str) -> dict[str, Any]:
    """A pose row as written to poses.csv (strings) or kept in memory, with typed fields."""
    out = dict(row)
    out["run"] = run
    out["pose_index"] = int(row["pose_index"])
    out["success"] = _flag(row.get("success"))
    out["accepted"] = _flag(row.get("accepted"))
    out["dissociated"] = _flag(row.get("dissociated"))
    out["energy_eh"] = _float(row.get("energy_eh"))
    out["li_coordination"] = _int(row.get("li_coordination"))
    return out


def assess(label: str, rows: list[dict[str, Any]], is_cluster: bool, tol: float = TOL_EV,
           crest_runs: Iterable[str] = ()) -> ClusterConvergence:
    """Convergence of one cluster from its pose rows, given in sampling order
    (run after run, pose_index within a run). Rows need run, pose_index, success,
    accepted, dissociated, energy_eh, calculation_id, li_coordination.
    `crest_runs`: tags of runs whose poses come from CREST."""
    crest_runs = set(crest_runs)
    crest_rows = [r for r in rows if r["run"] in crest_runs]
    if crest_rows:
        return _assess_with_crest(label, rows, crest_rows, is_cluster, tol)
    ok = [r for r in rows if r["success"]]
    acc = [r for r in ok if r["accepted"] and r["energy_eh"] is not None]
    runs = list(dict.fromkeys(r["run"] for r in rows))
    n_dis = sum(bool(r.get("dissociated")) for r in ok)
    reasons: list[str] = []
    if not acc:
        return ClusterConvergence(label, is_cluster, len(runs), len(rows), len(ok), 0, n_dis, 0, None, None, None,
                                  None, None, None, None, None, None, False, ["no_accepted_pose"], {})
    best = min(acc, key=lambda r: r["energy_eh"])
    e0 = best["energy_eh"]
    energies = [r["energy_eh"] for r in acc]
    near = sum((e - e0) * EH_TO_EV <= NEAR_BEST_EV for e in energies)
    first_half = acc[: math.ceil(len(acc) / 2)]
    half_drop = (min(r["energy_eh"] for r in first_half) - e0) * EH_TO_EV
    run_min: dict[str, float] = {}
    for r in acc:
        run_min[r["run"]] = min(run_min.get(r["run"], math.inf), r["energy_eh"])
    spread = (max(run_min.values()) - min(run_min.values())) * EH_TO_EV if len(run_min) > 1 else None
    if len(runs) > 1 and len(run_min) < len(runs):
        reasons.append("a_run_has_no_accepted_pose")
    drift = max(half_drop, spread or 0.0)
    if half_drop > tol:
        reasons.append("minimum_still_dropping")
    if spread is not None and spread > tol:
        reasons.append("runs_disagree")
    if is_cluster and len(rows) > 1 and near < 2:
        reasons.append("single_pose_minimum")
    if len(acc) < 2 and len(rows) > 1:
        reasons.append("fewer_than_two_accepted_poses")
    return ClusterConvergence(
        label=label, is_cluster=is_cluster, n_runs=len(runs), n_poses=len(rows), n_succeeded=len(ok),
        n_accepted=len(acc), n_dissociated=n_dis, n_near_best=near, best_energy_eh=e0, best_run=best["run"],
        best_pose=best["pose_index"], best_calculation_id=best.get("calculation_id"),
        best_li_coordination=best.get("li_coordination"),
        energy_range_eV=(max(energies) - min(energies)) * EH_TO_EV, half_drop_eV=half_drop, run_spread_eV=spread,
        sampling_drift_eV=drift, converged=not reasons, reasons=reasons, run_minima_eh=run_min)


def _assess_with_crest(label, rows, crest_rows, is_cluster, tol) -> ClusterConvergence:
    other = [r for r in rows if r not in crest_rows]
    base = assess(label, other, is_cluster, tol) if other else None
    full = assess(label, rows, is_cluster, tol)   # pooled best, counts, per-run minima
    crest_acc = [r for r in crest_rows if r["success"] and r["accepted"] and r["energy_eh"] is not None]
    full.crest_checked = True
    if not crest_acc:
        full.converged = False
        full.reasons = ["crest_no_accepted_pose"] + [x for x in full.reasons if x == "no_accepted_pose"]
        return full
    e_crest = min(r["energy_eh"] for r in crest_acc)
    e_base = base.best_energy_eh if base and base.best_energy_eh is not None else None
    gain = max(0.0, (e_base - e_crest) * EH_TO_EV) if e_base is not None else 0.0
    full.crest_gain_eV = gain
    full.half_drop_eV = None
    full.sampling_drift_eV = gain
    full.converged, full.reasons = True, []
    return full


def reaction_table(plan: Iterable[tuple[Any, str, Reaction]], minima: dict[str, dict],
                   conv: dict[str, ClusterConvergence]) -> list[dict[str, Any]]:
    """One row per (solvent, reaction). `minima`: label -> calculation record of the
    best pose; `conv`: label -> ClusterConvergence."""
    rows = []
    for solvent, name, rx in plan:
        row: dict[str, Any] = {"solvent": solvent.name, "smiles": solvent.canonical_smiles, "role": solvent.role,
                               "n_donor_atoms": solvent.n_donor_atoms, "reaction": name, "equation": str(rx)}
        try:
            e = reaction_energy(rx, minima)
            terms = [(c.label, nu, conv[c.label]) for c, nu in rx.terms()]
            flags = []
            for lab, _nu, c in terms:
                if c.is_cluster and c.n_poses > 1 and c.n_near_best < 2:
                    flags.append(f"single_pose_minimum:{lab}")
                if c.is_cluster and c.n_dissociated and c.n_dissociated >= c.n_poses / 2:
                    flags.append(f"ligand_detaches:{lab}")
                if any(x != "single_pose_minimum" for x in c.reasons):
                    flags.append(f"not_converged:{lab}")
            drift = sum(abs(nu) * (c.sampling_drift_eV or 0.0) for _lab, nu, c in terms)
            row.update({"delta_e_eV": round(e["delta_e_eV"], 4),
                        "sampling_drift_eV": round(drift, 4),
                        "delta_e_kcal_mol": round(e["delta_e_kcal_mol"], 3),
                        "method": e["method"], "solvation": f"{e['solvation_model']}({e['solvation_reference']})",
                        "min_accepted_poses": min(c.n_accepted for _l, _n, c in terms),
                        "min_poses_near_best": min(c.n_near_best for _l, _n, c in terms),
                        "all_terms_converged": all(c.converged for _l, _n, c in terms),
                        "flags": ";".join(flags),
                        "status": "ok" if not flags else "ok_with_flags"})
        except MissingEnergyError as err:
            row.update({"status": f"missing: {err}"})
        except InconsistentEnergiesError as err:
            row.update({"status": f"inconsistent: {err}"})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def print_reactions(rows: list[dict]) -> None:
    def fmt(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) else "-".rjust(int(spec.split(".")[0]))

    print()
    print(f"{'solvent':8s} {'reaction':16s} {'dE (eV)':>9s} {'drift':>7s} {'dE (kcal/mol)':>14s}  status / flags")
    for r in rows:
        flags = [f.split(":")[0] for f in (r.get("flags") or "").split(";") if f]
        note = ", ".join(f"{flags.count(k)}x {k}" for k in dict.fromkeys(flags)) or r["status"]
        print(f"{r['solvent']:8s} {r['reaction']:16s} {fmt(r.get('delta_e_eV'), '9.3f')} "
              f"{fmt(r.get('sampling_drift_eV'), '7.3f')} {fmt(r.get('delta_e_kcal_mol'), '14.2f')}  {note[:90]}")
    print("\ndrift: how much the minima of the terms moved with more poses or between runs (eV, summed);")
    print("  a lower bound of the sampling error. Differences between solvents smaller than it are noise.")
    print("single_pose_minimum / not_converged: send the cluster to CREST (merge_runs --crest-export).")
    print("ligand_detaches: in most poses a ligand left Li+; the cluster is not a stable motif in this continuum.")
