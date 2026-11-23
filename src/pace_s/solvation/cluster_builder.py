"""Initial 3D geometries (poses) of Li+ / FSI- / solvent clusters.

Li+ sits at the origin. Each ligand (solvent molecule or FSI-) is attached by
one *binding mode*: one donor atom (monodentate) or two donor atoms whose
distance in the chosen conformer allows chelation (2.4-3.4 A, e.g. the two O
of a gauche DME, or one O on each S of a cis FSI-). The ligand is rotated so
its body points away from Li+ along an assigned direction (linear, trigonal,
tetrahedral, ... for 2, 3, 4 ligands) and spun about that axis at random.
Poses differ in conformers, binding modes and spins, all drawn from a seeded
generator: the same seed gives the same poses.

Pose 0 is the "chemist's guess": the lowest-energy force-field conformer that
offers a chelating mode, if any, for every ligand. The remaining poses are
random. Clashes are resolved by redrawing, then by pushing ligands outward.

The geometries are starting points for xtb; nothing here is an energy minimum.
Coordination actually reached after optimization is measured in `ensemble`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from pace_s.descriptors.geometry import Structure, lone_pair_direction, rotation_aligning
from pace_s.solvation.species import Composition, Species

LI_DONOR_DISTANCE_A = {"O": 2.0, "N": 2.1, "S": 2.45}
CHELATE_RANGE_A = (2.4, 3.4)
MIN_INTERLIGAND_A = 1.9   # any two atoms of different ligands
MIN_TO_LI_A = 1.75        # non-donor atoms to Li+


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class BindingMode:
    donors: tuple[int, ...]   # atom indices in the ligand
    point: tuple[float, float, float]  # where Li+ sits, in ligand coordinates

    @property
    def denticity(self) -> int:
        return len(self.donors)


@dataclass
class LigandConformer:
    structure: Structure
    ff_energy: float
    modes: list[BindingMode]


@dataclass
class LigandTemplate:
    species: Species
    mol: Chem.Mol
    conformers: list[LigandConformer]


@dataclass
class Pose:
    structure: Structure
    pose_index: int
    ligand_slices: list[tuple[int, int]]  # atom ranges of each ligand in `structure` (Li is atom 0)
    ligand_names: list[str]
    binding: list[dict] = field(default_factory=list)  # per ligand: conformer, donors, denticity


def binding_donor_atoms(species: Species, mol: Chem.Mol) -> list[int]:
    """Donors used to place Li+. Anions also bind through charged atoms (the N of FSI-)."""
    if species.role == "anion":
        return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in ("O", "N", "S") and
                not (a.GetSymbol() == "S" and a.GetDegree() > 2)]
    return species.donor_atoms()


def _binding_modes(struct: Structure, mol: Chem.Mol, donors: list[int]) -> list[BindingMode]:
    modes: list[BindingMode] = []
    for d in donors:
        dist = LI_DONOR_DISTANCE_A.get(struct.symbols[d], 2.0)
        p = struct.coords[d] + dist * lone_pair_direction(struct, mol, d)
        modes.append(BindingMode((d,), tuple(p)))
    lo, hi = CHELATE_RANGE_A
    for i_pos, i in enumerate(donors):
        for j in donors[i_pos + 1:]:
            rij = struct.coords[j] - struct.coords[i]
            dij = float(np.linalg.norm(rij))
            if not lo <= dij <= hi:
                continue
            d = 0.5 * (LI_DONOR_DISTANCE_A.get(struct.symbols[i], 2.0) + LI_DONOR_DISTANCE_A.get(struct.symbols[j], 2.0))
            h = np.sqrt(max(d * d - (dij / 2) ** 2, 0.64))
            out = lone_pair_direction(struct, mol, i) + lone_pair_direction(struct, mol, j)
            out -= np.dot(out, rij) / dij**2 * rij  # perpendicular to the O...O axis
            if np.linalg.norm(out) < 1e-6:
                continue
            out /= np.linalg.norm(out)
            p = 0.5 * (struct.coords[i] + struct.coords[j]) + h * out
            # reject a chelation point buried inside the ligand
            others = np.delete(struct.coords, [i, j], axis=0)
            if len(others) and np.min(np.linalg.norm(others - p, axis=1)) < MIN_TO_LI_A:
                continue
            modes.append(BindingMode((i, j), tuple(p)))
    return modes


@lru_cache(maxsize=256)
def _template(species: Species, n_conformers: int, seed: int) -> LigandTemplate:
    mol = species.mol
    if mol.GetNumAtoms() == 1:
        raise BuildError("monatomic species are not ligands")
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.3
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not ids:
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not ids:
        raise BuildError(f"RDKit could not embed {species.name}")
    props = AllChem.MMFFGetMoleculeProperties(mol)
    donors = binding_donor_atoms(species, mol)
    if not donors:
        raise BuildError(f"{species.name} has no donor atoms")
    confs = []
    for cid in ids:
        energy = 0.0
        try:
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            ff.Minimize(maxIts=2000)
            energy = float(ff.CalcEnergy())
        except Exception:  # noqa: BLE001 - no force field for this species: keep ETKDG geometry
            pass
        st = Structure.from_mol(mol, cid)
        confs.append(LigandConformer(st, energy, _binding_modes(st, mol, donors)))
    confs.sort(key=lambda c: c.ff_energy)
    return LigandTemplate(species, mol, confs)


def ligand_template(species: Species, n_conformers: int = 10, seed: int = 20261031) -> LigandTemplate:
    return _template(species, n_conformers, seed)


def directions(k: int) -> np.ndarray:
    """k unit vectors spread around Li+ (linear, trigonal, tetrahedral, then Fibonacci sphere)."""
    if k == 1:
        return np.array([[0.0, 0.0, 1.0]])
    if k == 2:
        return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    if k == 3:
        a = np.deg2rad([90.0, 210.0, 330.0])
        return np.stack([np.cos(a), np.sin(a), np.zeros(3)], axis=1)
    if k == 4:
        v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
        return v / np.sqrt(3)
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _orient(conf: LigandConformer, mode: BindingMode, direction: np.ndarray, spin: float) -> Structure:
    """Ligand coordinates with its binding point at the origin and its body along `direction`."""
    p = np.asarray(mode.point)
    body = conf.structure.coords.mean(axis=0) - p
    st = conf.structure.translated(-p)
    if np.linalg.norm(body) > 1e-6:
        st = st.rotated(rotation_aligning(body, direction), about=np.zeros(3))
    return st.rotated(_axis_rotation(direction, spin), about=np.zeros(3))


def _violations(ligands: list[Structure], donor_sets: list[tuple[int, ...]]) -> np.ndarray:
    """Per-ligand worst violation (A) of the distance rules; zeros if none."""
    v = np.zeros(len(ligands))
    for a in range(len(ligands)):
        d_li = np.linalg.norm(ligands[a].coords, axis=1)
        mask = np.ones(len(d_li), bool)
        mask[list(donor_sets[a])] = False
        if mask.any():
            v[a] = max(v[a], MIN_TO_LI_A - float(d_li[mask].min()))
        for b in range(a + 1, len(ligands)):
            x = MIN_INTERLIGAND_A - ligands[a].min_distance_to(ligands[b])
            v[a], v[b] = max(v[a], x), max(v[b], x)
    return np.clip(v, 0.0, None)


def _clash(ligands: list[Structure], donor_sets: list[tuple[int, ...]]) -> float:
    """Worst violation (A) of the distance rules; 0 if none."""
    return float(_violations(ligands, donor_sets).max()) if ligands else 0.0


def _ligand_list(comp: Composition) -> list[Species]:
    out: list[Species] = []
    for sp, n in comp.parts:
        if sp.role == "li_cation":
            continue
        out += [sp] * n
    return out


def build_poses(comp: Composition, n_poses: int = 6, seed: int = 20261031, n_conformers: int = 10,
                max_tries: int = 60) -> list[Pose]:
    """Up to `n_poses` initial geometries of a composition.

    Isolated species (no Li+): distinct force-field conformers, lowest first
    (a single atom gives one pose). Clusters with one Li+: ligands placed as
    described in the module docstring.
    """
    n_li = comp.count("li_cation")
    ligands = _ligand_list(comp)
    if n_li == 0:
        if len(ligands) != 1:
            raise BuildError("isolated compositions must contain exactly one molecule")
        sp = ligands[0]
        tpl = ligand_template(sp, n_conformers, seed)
        poses = []
        for k, conf in enumerate(tpl.conformers[:n_poses]):
            poses.append(Pose(conf.structure, k, [(0, len(conf.structure.symbols))], [sp.name],
                              [{"conformer": k, "donors": [], "denticity": 0}]))
        return poses
    if n_li != 1:
        raise BuildError("only clusters with one Li+ are supported")
    li = Structure(["Li"], np.zeros((1, 3)))
    if not ligands:
        return [Pose(li, 0, [], [], [])]

    templates = [ligand_template(sp, n_conformers, seed) for sp in ligands]
    dirs = directions(len(ligands))
    rng = np.random.default_rng(seed)
    poses: list[Pose] = []
    seen: set[tuple] = set()
    attempts = 0
    while len(poses) < n_poses and attempts < n_poses * 4:
        attempts += 1
        best = None
        guess = not poses  # pose 0: the chemist's guess, kept even if it needs pushing apart
        for _ in range(1 if guess else max_tries):
            placed, donor_sets, binding, choice_key = [], [], [], []
            for t, tpl in enumerate(templates):
                if guess:
                    # pose 0: lowest conformer with the highest-denticity mode
                    ci = next((k for k, c in enumerate(tpl.conformers) if any(m.denticity > 1 for m in c.modes)), 0)
                    conf = tpl.conformers[ci]
                    mode = max(conf.modes, key=lambda m: m.denticity)
                    spin = 0.0
                else:
                    ci = int(rng.integers(len(tpl.conformers)))
                    conf = tpl.conformers[ci]
                    mode = conf.modes[int(rng.integers(len(conf.modes)))]
                    spin = float(rng.uniform(0, 2 * np.pi))
                placed.append(_orient(conf, mode, dirs[t], spin))
                donor_sets.append(mode.donors)
                binding.append({"conformer": ci, "donors": list(mode.donors), "denticity": mode.denticity})
                choice_key.append((ci, mode.donors))
            v = _clash(placed, donor_sets)
            if best is None or v < best[0]:
                best = (v, placed, donor_sets, binding, tuple(choice_key))
            if v == 0.0:
                break
        v, placed, donor_sets, binding, key = best
        for _ in range(30):  # push only the clashing ligands outward until the clash is gone
            if v == 0.0:
                break
            viol = _violations(placed, donor_sets)
            placed = [p.translated(0.1 * dirs[t]) if viol[t] > 0 else p for t, p in enumerate(placed)]
            v = _clash(placed, donor_sets)
        if key in seen and len(poses) > 0:
            continue
        seen.add(key)
        st, slices, start = li, [], 1
        for p in placed:
            st = st + p
            slices.append((start, start + len(p.symbols)))
            start += len(p.symbols)
        poses.append(Pose(st, len(poses), slices, [sp.name for sp in ligands], binding))
    if not poses:
        raise BuildError(f"no pose could be built for {comp.label}")
    return poses
