"""Geometries for tier 0: RDKit (or CREST) conformers and placement of
Li+ / Li2S8 on a molecule's coordination sites.

Convention: the atom order of the RDKit `Mol` (with explicit H) is the same
as in the xyz written out and in the optimized xyz xtb returns, so site
indices computed on the graph are valid on the optimized coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

COORDINATING_ELEMENTS = ("O", "N", "S")
ION_DISTANCE_A = {"O": 1.95, "N": 2.05, "S": 2.45}  # initial Li–X distance; xtb relaxes it


@dataclass
class Structure:
    symbols: list[str]
    coords: np.ndarray  # (n, 3) Å

    @classmethod
    def from_xyz(cls, path: str | Path) -> "Structure":
        lines = Path(path).read_text().splitlines()
        n = int(lines[0].split()[0])
        symbols, coords = [], []
        for line in lines[2 : 2 + n]:
            parts = line.split()
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
        return cls(symbols, np.asarray(coords, dtype=float))

    @classmethod
    def from_mol(cls, mol: Chem.Mol, conf_id: int = -1) -> "Structure":
        conf = mol.GetConformer(conf_id)
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        return cls([a.GetSymbol() for a in mol.GetAtoms()], coords)

    def write_xyz(self, path: str | Path, comment: str = "") -> Path:
        path = Path(path)
        lines = [str(len(self.symbols)), comment]
        lines += [f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}" for s, (x, y, z) in zip(self.symbols, self.coords)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def centroid(self) -> np.ndarray:
        return self.coords.mean(axis=0)

    def translated(self, shift: np.ndarray) -> "Structure":
        return Structure(list(self.symbols), self.coords + shift)

    def rotated(self, rot: np.ndarray, about: np.ndarray) -> "Structure":
        return Structure(list(self.symbols), (self.coords - about) @ rot.T + about)

    def __add__(self, other: "Structure") -> "Structure":
        return Structure(self.symbols + other.symbols, np.vstack([self.coords, other.coords]))

    def min_distance_to(self, other: "Structure") -> float:
        d = np.linalg.norm(self.coords[:, None, :] - other.coords[None, :, :], axis=-1)
        return float(d.min())


def mol_with_hs(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.AddHs(mol)


def embed_lowest(mol: Chem.Mol, n_conformers: int = 10, seed: int = 20260924) -> tuple[Chem.Mol, int, int]:
    """Embed `n_conformers` with ETKDG, optimize with MMFF94 and return (mol,
    id of the lowest-energy conformer, number of conformers generated)."""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.3
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not ids:
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not ids:
        raise ValueError("RDKit could not embed the molecule")
    props = AllChem.MMFFGetMoleculeProperties(mol)
    energies = []
    for cid in ids:
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            ff.Minimize(maxIts=2000)
            energies.append(ff.CalcEnergy())
        else:  # MMFF does not parametrize e.g. Si: UFF as fallback
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            ff.Minimize(maxIts=2000)
            energies.append(ff.CalcEnergy())
    best = ids[int(np.argmin(energies))]
    return mol, best, len(ids)


def coordination_sites(mol: Chem.Mol, charges: list[float] | None = None, max_sites: int = 4) -> list[int]:
    """Indices of coordinating heteroatoms (O, N, S with a lone pair), ordered by
    most negative Mulliken charge if charges are given, otherwise by atom index."""
    sites = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() in COORDINATING_ELEMENTS and a.GetTotalNumHs() == 0 and not a.GetIsAromatic()
    ]
    if charges and len(charges) == mol.GetNumAtoms():
        sites.sort(key=lambda i: charges[i])
    return sites[:max_sites]


def lone_pair_direction(struct: Structure, mol: Chem.Mol, site: int) -> np.ndarray:
    """Unit vector from the site, opposite to the centroid of its neighbors."""
    neigh = [n.GetIdx() for n in mol.GetAtomWithIdx(site).GetNeighbors()]
    if not neigh:
        return np.array([1.0, 0.0, 0.0])
    v = struct.coords[site] - struct.coords[neigh].mean(axis=0)
    norm = np.linalg.norm(v)
    if norm < 1e-6:  # linear site (e.g. a perfectly symmetric ether O): take a perpendicular
        a = struct.coords[neigh[0]] - struct.coords[site]
        v = np.cross(a, [0.0, 0.0, 1.0])
        if np.linalg.norm(v) < 1e-6:
            v = np.cross(a, [0.0, 1.0, 0.0])
        norm = np.linalg.norm(v)
    return v / norm


def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector `a` onto `b` (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            return np.eye(3)
        # antiparallel: rotate 180 degrees about any perpendicular axis
        p = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(p) < 1e-8:
            p = np.cross(a, [0.0, 1.0, 0.0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def place_ion(struct: Structure, mol: Chem.Mol, site: int, symbol: str = "Li") -> Structure:
    """Molecule + monatomic ion placed along the lone pair of the site."""
    d = ION_DISTANCE_A.get(struct.symbols[site], 2.0)
    pos = struct.coords[site] + d * lone_pair_direction(struct, mol, site)
    return struct + Structure([symbol], pos[None, :])


def place_cluster(
    struct: Structure,
    mol: Chem.Mol,
    site: int,
    cluster: Structure,
    anchor: int,
    min_separation: float = 2.2,
) -> Structure:
    """Molecule + cluster (e.g. Li2S8), with the cluster's `anchor` atom on the
    site's lone pair and the rest of the cluster pointing outwards. If any
    contact is closer than `min_separation` Å, the cluster is pushed away along
    the lone-pair direction."""
    direction = lone_pair_direction(struct, mol, site)
    d = ION_DISTANCE_A.get(struct.symbols[site], 2.0)
    target = struct.coords[site] + d * direction
    # orient: anchor -> cluster centroid vector aligned with `direction`
    out = cluster.centroid() - cluster.coords[anchor]
    if np.linalg.norm(out) > 1e-6:
        rot = rotation_aligning(out, direction)
        cluster = cluster.rotated(rot, about=cluster.coords[anchor])
    cluster = cluster.translated(target - cluster.coords[anchor])
    for _ in range(20):
        if struct.min_distance_to(cluster) >= min_separation * 0.8:
            break
        cluster = cluster.translated(0.25 * direction)
    return struct + cluster
