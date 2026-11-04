"""Quality control of optimized cluster poses and choice of the minimum.

After xtb optimizes a pose, three things are checked:

- connectivity: covalent bonds inside every ligand (distance-based, Li excluded)
  must be the same before and after. A change (decomposition, proton or F
  transfer, ring opening) flags the pose as reactive; it is never used as a
  solvation minimum.
- attachment: every ligand must keep at least one atom within 3.0 A of Li+.
  A ligand that drifted away means the cluster dissociated in the continuum;
  the pose is flagged as dissociated and not used.
- coordination: number of donor atoms (O, N, S) within 2.6 A of Li+ and of F
  atoms within 2.4 A (fluorinated diluents can hold Li+ through F instead of
  O); reported, not filters.

`select_minimum` takes the lowest-energy pose among those that pass and
reports how many accepted poses reached it (within 0.05 eV) and how many
dissociated. A minimum found by a single pose, or a cluster whose ligands
mostly detach, is flagged: the first is a candidate for CREST, the second
means the cluster is not a stable solvation motif in this continuum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from rdkit import Chem

from pace_s.descriptors.geometry import Structure

COVALENT_SCALE = 1.25
ATTACHED_A = 3.0
COORDINATION_A = 2.6
LI_F_CONTACT_A = 2.4
NEAR_BEST_EV = 0.05
_PT = Chem.GetPeriodicTable()


def bonds(struct: Structure, atoms: Iterable[int] | None = None) -> set[tuple[int, int]]:
    """Distance-based covalent bonds (r < 1.25 x sum of covalent radii), Li excluded."""
    idx = [i for i in (atoms if atoms is not None else range(len(struct.symbols))) if struct.symbols[i] != "Li"]
    radii = np.array([_PT.GetRcovalent(struct.symbols[i]) for i in idx])
    xyz = struct.coords[idx]
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    cut = COVALENT_SCALE * (radii[:, None] + radii[None, :])
    out = set()
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if d[a, b] < cut[a, b]:
                out.add((idx[a], idx[b]))
    return out


@dataclass
class PoseQC:
    connectivity_preserved: bool
    dissociated: bool
    li_coordination: int | None
    li_f_contacts: int | None = None
    ligand_min_li_distance_A: list[float] = field(default_factory=list)
    bonds_broken: list[tuple[int, int]] = field(default_factory=list)
    bonds_formed: list[tuple[int, int]] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.connectivity_preserved and not self.dissociated


def check_pose(initial: Structure, final: Structure, ligand_slices: list[tuple[int, int]],
               donor_atoms: list[int] | None = None) -> PoseQC:
    """Compare initial and optimized geometries of one pose (same atom order).
    `donor_atoms` are global indices of donor atoms (for the coordination count)."""
    if initial.symbols != final.symbols:
        raise ValueError("initial and final structures have different atoms")
    b0, b1 = bonds(initial), bonds(final)
    broken, formed = sorted(b0 - b1), sorted(b1 - b0)
    # bonds between different ligands count as "formed" too (bonds() sees all non-Li atoms)
    has_li = "Li" in final.symbols
    min_d, dissociated, cn, nf = [], False, None, None
    if has_li and ligand_slices:
        li = final.coords[final.symbols.index("Li")]
        for lo, hi in ligand_slices:
            dmin = float(np.linalg.norm(final.coords[lo:hi] - li, axis=1).min())
            min_d.append(round(dmin, 3))
            dissociated |= dmin > ATTACHED_A
        if donor_atoms:
            cn = int(sum(np.linalg.norm(final.coords[i] - li) < COORDINATION_A for i in donor_atoms))
        f_idx = [i for i, s in enumerate(final.symbols) if s == "F"]
        nf = int(sum(np.linalg.norm(final.coords[i] - li) < LI_F_CONTACT_A for i in f_idx))
    return PoseQC(not broken and not formed, dissociated, cn, nf, min_d, broken, formed)


@dataclass
class EnsembleSummary:
    label: str
    n_poses: int
    n_succeeded: int
    n_accepted: int
    best_pose: int | None
    best_energy_eh: float | None
    energy_range_eV: float | None  # max - min energy among accepted poses
    best_li_coordination: int | None
    n_near_best: int = 0       # accepted poses within NEAR_BEST_EV of the best
    n_dissociated: int = 0


def select_minimum(label: str, rows: list[dict]) -> EnsembleSummary:
    """rows: one per pose with keys pose_index, success, energy_eh, accepted, li_coordination."""
    from pace_s.descriptors.xtb import EH_TO_EV

    ok = [r for r in rows if r.get("success")]
    acc = [r for r in ok if r.get("accepted")]
    n_dis = sum(bool(r.get("dissociated")) for r in ok)
    if not acc:
        return EnsembleSummary(label, len(rows), len(ok), 0, None, None, None, None, 0, n_dis)
    best = min(acc, key=lambda r: r["energy_eh"])
    energies = [r["energy_eh"] for r in acc]
    near = sum((e - best["energy_eh"]) * EH_TO_EV <= NEAR_BEST_EV for e in energies)
    return EnsembleSummary(label, len(rows), len(ok), len(acc), best["pose_index"], best["energy_eh"],
                           (max(energies) - min(energies)) * EH_TO_EV, best.get("li_coordination"), near, n_dis)
