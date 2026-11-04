"""Tier 0.5 geometry: initial poses, binding modes, reproducibility, QC of optimized poses."""

import numpy as np
import pytest

from pace_s.descriptors.geometry import Structure
from pace_s.solvation.cluster_builder import (
    MIN_INTERLIGAND_A,
    build_poses,
    directions,
    ligand_template,
)
from pace_s.solvation.ensemble import bonds, check_pose, select_minimum
from pace_s.solvation.species import Species, build_composition, fsi_anion

DME = Species("DME", "COCCOC", "solvent")
THP = Species("THP", "C1CCOCC1", "solvent")
TTE = Species("TTE", "FC(F)C(F)(F)OCC(F)(F)C(F)F", "diluent")


def test_directions_are_unit_and_spread():
    for k in range(1, 7):
        d = directions(k)
        assert d.shape == (k, 3)
        assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
        if 1 < k <= 4:
            cos = d @ d.T
            assert cos[~np.eye(k, dtype=bool)].max() < 0.1


def test_dme_has_a_chelating_mode_and_fsi_binds_through_n():
    dme = ligand_template(DME)
    assert any(m.denticity == 2 for c in dme.conformers for m in c.modes)
    fsi = ligand_template(fsi_anion())
    n_idx = [a.GetIdx() for a in fsi.mol.GetAtoms() if a.GetSymbol() == "N"][0]
    modes = [m for c in fsi.conformers for m in c.modes]
    assert any(m.donors == (n_idx,) for m in modes)
    assert any(m.denticity == 2 for m in modes)  # O,O (same or different S)


@pytest.mark.parametrize("protocol, solvent", [("li_s1", DME), ("li_s2", DME), ("li_fsi", None),
                                               ("li_fsi_s1", DME), ("li_sat", THP), ("li_fsi_sat", TTE)])
def test_poses_have_the_right_atoms_and_no_clashes(protocol, solvent):
    comp = build_composition(protocol, solvent)
    poses = build_poses(comp, n_poses=4, seed=7)
    assert poses
    n_atoms = sum(sp.mol.GetNumAtoms() * n for sp, n in comp.parts)
    for p in poses:
        st = p.structure
        assert len(st.symbols) == n_atoms
        assert st.symbols[0] == "Li" and np.allclose(st.coords[0], 0.0)
        assert len(p.ligand_slices) == comp.n_molecules - 1
        ligs = [Structure(st.symbols[lo:hi], st.coords[lo:hi]) for lo, hi in p.ligand_slices]
        for a in range(len(ligs)):
            for b in range(a + 1, len(ligs)):
                assert ligs[a].min_distance_to(ligs[b]) >= MIN_INTERLIGAND_A - 1e-6
        # every ligand points at Li+; crowded clusters (4 TTE) are pushed apart, so only a loose bound
        for lig in ligs:
            assert np.linalg.norm(lig.coords, axis=1).min() < 4.5


def test_uncrowded_saturated_cluster_keeps_donors_at_bonding_distance():
    for pose in build_poses(build_composition("li_sat", THP), n_poses=4, seed=7):
        st = pose.structure
        for lo, hi in pose.ligand_slices:
            assert np.linalg.norm(st.coords[lo:hi], axis=1).min() < 3.0  # at most a small push (RDKit-version dependent)


def test_poses_are_reproducible():
    comp = build_composition("li_fsi_s1", DME)
    a = build_poses(comp, n_poses=3, seed=11)
    b = build_poses(comp, n_poses=3, seed=11)
    assert all(np.allclose(x.structure.coords, y.structure.coords) for x, y in zip(a, b))


def test_isolated_species_give_conformers():
    poses = build_poses(build_composition("isolated_solvent", DME), n_poses=3)
    assert 1 <= len(poses) <= 3 and all(p.ligand_slices == [(0, 16)] for p in poses)
    li = build_poses(build_composition("isolated_li"), n_poses=5)
    assert len(li) == 1 and li[0].structure.symbols == ["Li"]


def test_qc_detects_broken_bond_and_dissociation():
    comp = build_composition("li_s1", DME)
    pose = build_poses(comp, n_poses=1)[0]
    st = pose.structure
    ok = check_pose(st, st, pose.ligand_slices, [1 + 1, 1 + 4])
    assert ok.accepted and ok.li_coordination is not None and ok.li_f_contacts == 0
    # break the first C-O bond of the ligand
    broken = Structure(list(st.symbols), st.coords.copy())
    c, o = sorted(next(iter(bonds(st))))
    broken.coords[o] += 3.0 * (broken.coords[o] - broken.coords[c]) / np.linalg.norm(broken.coords[o] - broken.coords[c])
    qc = check_pose(st, broken, pose.ligand_slices)
    assert not qc.connectivity_preserved and not qc.accepted
    # move the whole ligand 8 A away from Li+ along its body direction
    far = Structure(list(st.symbols), st.coords.copy())
    lo, hi = pose.ligand_slices[0]
    u = far.coords[lo:hi].mean(axis=0) / np.linalg.norm(far.coords[lo:hi].mean(axis=0))
    far.coords[lo:hi] += 8.0 * u
    qc = check_pose(st, far, pose.ligand_slices)
    assert qc.dissociated and qc.connectivity_preserved and not qc.accepted


def test_select_minimum_ignores_rejected_and_counts_near_best():
    rows = [
        {"pose_index": 0, "success": True, "energy_eh": -10.000, "accepted": True},
        {"pose_index": 1, "success": True, "energy_eh": -10.001, "accepted": True},
        {"pose_index": 2, "success": True, "energy_eh": -10.500, "accepted": False, "dissociated": True},
        {"pose_index": 3, "success": False, "energy_eh": None, "accepted": False},
    ]
    s = select_minimum("X", rows)
    assert s.best_pose == 1 and s.n_accepted == 2 and s.n_near_best == 2 and s.n_dissociated == 1
    none = select_minimum("Y", rows[2:])
    assert none.best_pose is None and none.best_energy_eh is None
