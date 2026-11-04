"""Run xtb on cluster poses and write cluster / calculation records.

One pose = one working directory `<workdir>/<composition label>/pose_XX`. The
runner writes `input.xyz`, calls `pace_s.descriptors.xtb.run_xtb`, checks the
optimized geometry (`ensemble.check_pose`) and stores `result.json` with the
calculation record and the QC. If `result.json` already exists with the same
input hash and a successful result, the pose is not recomputed (resumable).

Only `input.xyz`, `xtbopt.xyz`, `xtb.out` and `result.json` are kept; the rest
of what xtb writes is deleted (keeps the number of files low on Grace scratch).

Failures are records too: success=false, energy_eh=null and an error_code.
"""

from __future__ import annotations

import json
import re
import socket
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pace_s.contracts import errors as contract_errors
from pace_s.descriptors.geometry import Structure
from pace_s.descriptors.xtb import XtbError, run_xtb
from pace_s.provenance import file_sha256, stable_hash
from pace_s.solvation.cluster_builder import Pose, binding_donor_atoms
from pace_s.solvation.ensemble import check_pose
from pace_s.solvation.species import Composition

KEEP = {"input.xyz", "xtbopt.xyz", "xtb.out", "result.json"}
_RE_VERSION = re.compile(r"xtb version\s+([0-9][0-9.]*)")


@dataclass
class XtbSettings:
    gfn: int = 2
    alpb: str | None = "ether"
    threads: int = 4
    timeout_s: int = 1800

    @property
    def method(self) -> str:
        return f"GFN{self.gfn}-xTB"

    @property
    def solvation_model(self) -> str:
        return "ALPB" if self.alpb else "none"


def _donor_indices(comp: Composition, pose: Pose) -> list[int]:
    """Global indices of the donor atoms of every ligand in the pose."""
    out = []
    ligands = [sp for sp, n in comp.parts if sp.role != "li_cation" for _ in range(n)]
    for sp, (lo, _hi) in zip(ligands, pose.ligand_slices):
        out += [lo + i for i in binding_donor_atoms(sp, sp.mol)]
    return out


def _error_code(message: str) -> str:
    m = message.lower()
    if "exceeded" in m and " s in " in m:
        return "TIMEOUT"
    if "failed to converge" in m or "geometry optimization" in m and "converge" in m:
        return "OPT_NOT_CONVERGED"
    if "scf" in m and "converge" in m:
        return "SCF_NOT_CONVERGED"
    if "not found" in m:
        return "UNSUPPORTED_METHOD"
    return "PROGRAM_ERROR"


def _cleanup(d: Path) -> None:
    for f in d.iterdir():
        if f.is_file() and f.name not in KEEP:
            try:
                f.unlink()
            except OSError:
                pass


def run_pose(
    comp: Composition,
    pose: Pose,
    workdir: Path,
    settings: XtbSettings,
    cluster_id: str,
    calculation_id: str,
    campaign_id: str,
    commit: str | None = None,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Optimize one pose; return {'calculation': record, 'qc': {...}, 'pose_index': k}.
    `runner` defaults to pace_s.descriptors.xtb.run_xtb (tests pass a mock)."""
    runner = runner or run_xtb
    d = Path(workdir)
    d.mkdir(parents=True, exist_ok=True)
    n_atoms = len(pose.structure.symbols)
    xyz = pose.structure.write_xyz(d / "input.xyz", f"{comp.label} pose {pose.pose_index}")
    inputs = {
        "symbols": pose.structure.symbols,
        "coords": [[round(float(x), 6) for x in row] for row in pose.structure.coords],
        "charge": comp.net_charge, "multiplicity": comp.multiplicity,
        "method": settings.method, "alpb": settings.alpb, "opt": n_atoms > 1,
    }
    ihash = stable_hash(inputs)
    result_file = d / "result.json"
    if result_file.exists():
        old = json.loads(result_file.read_text(encoding="utf-8"))
        if old["calculation"].get("input_hash") == ihash and old["calculation"].get("success"):
            # IDs are local to a run: a resumed pose takes the IDs of the current run
            rec = old["calculation"]
            rec.update({"calculation_id": calculation_id, "cluster_id": cluster_id, "campaign_id": campaign_id,
                        "geometry_id": f"{cluster_id}-{calculation_id}"})
            old["pose_index"] = pose.pose_index
            return old

    cmd = (f"xtb input.xyz --gfn {settings.gfn} --chrg {comp.net_charge}"
           + (f" --uhf {comp.multiplicity - 1}" if comp.multiplicity > 1 else "")
           + (" --opt" if n_atoms > 1 else "")
           + (f" --alpb {settings.alpb}" if settings.alpb else ""))
    rec: dict[str, Any] = {
        "schema_version": "1.0.0",
        "calculation_id": calculation_id,
        "cluster_id": cluster_id,
        "campaign_id": campaign_id,
        "method": settings.method,
        "program": {"name": "xtb"},
        "solvation_model": settings.solvation_model,
        "solvation_reference": settings.alpb if settings.alpb else None,
        # a single atom (Li+) has nothing to optimize; it is labelled like the other terms so
        # that reaction energies can combine it (parameters.opt records that no opt ran)
        "geometry_type": "xtb_optimized",
        "energy_kind": "electronic_total",
        "energy_eh": None,
        "charge": comp.net_charge,
        "multiplicity": comp.multiplicity,
        "success": False,
        "error_code": None,
        "input_hash": ihash,
        "command": cmd,
        "input_geometry_path": str(xyz),
        "n_threads": settings.threads,
        "host": socket.gethostname(),
        "parameters": {"gfn": settings.gfn, "alpb": settings.alpb, "opt": n_atoms > 1},
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if commit:
        rec["commit"] = commit
    qc: dict[str, Any] = {}
    t0 = time.time()
    try:
        res = runner(xyz, d, charge=comp.net_charge, uhf=comp.multiplicity - 1, opt=n_atoms > 1,
                     gfn=settings.gfn, alpb=settings.alpb, threads=settings.threads, timeout_s=settings.timeout_s)
        rec["runtime_s"] = round(time.time() - t0, 2)
        log = d / "xtb.out"
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        m = _RE_VERSION.search(text)
        rec["program"]["version"] = m.group(1) if m else "unknown"
        rec["log_path"] = str(log)
        rec["log_sha256"] = file_sha256(log) if log.exists() else "0" * 64
        final_xyz = Path(res.opt_xyz) if res.opt_xyz else xyz
        final = Structure.from_xyz(final_xyz)
        rec["final_geometry_path"] = str(final_xyz)
        rec["geometry_id"] = f"{cluster_id}-{calculation_id}"
        check = check_pose(pose.structure, final, pose.ligand_slices, _donor_indices(comp, pose))
        qc = asdict(check) | {"accepted": check.accepted}
        rec["connectivity_preserved"] = check.connectivity_preserved
        rec["energy_eh"] = float(res.energy_eh)
        rec["success"] = True
    except XtbError as e:
        rec["runtime_s"] = round(time.time() - t0, 2)
        rec["error_code"] = _error_code(str(e))
        rec["error_message"] = str(e)[-500:]
        rec["energy_eh"] = None
        log = d / "xtb.out"
        if log.exists():
            rec["log_path"] = str(log)
            rec["log_sha256"] = file_sha256(log)
            m = _RE_VERSION.search(log.read_text(encoding="utf-8", errors="replace"))
            if m:
                rec["program"]["version"] = m.group(1)
    problems = contract_errors("calculation_record", rec)
    if problems:
        raise ValueError(f"calculation record of {cluster_id} does not satisfy the contract: {problems}")
    out = {"pose_index": pose.pose_index, "calculation": rec, "qc": qc, "binding": pose.binding}
    result_file.write_text(json.dumps(out, indent=1, default=list), encoding="utf-8")
    _cleanup(d)
    return out
