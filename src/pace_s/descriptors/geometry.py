"""Geometrías para tier 0: confórmeros con RDKit (o CREST) y colocación de
Li+ / Li2S8 en los sitios de coordinación de una molécula.

Convención: el orden de átomos del `Mol` de RDKit (con H explícitos) es el
mismo que el del xyz que se escribe y que el que xtb devuelve optimizado, así
que los índices de sitio calculados sobre el grafo valen sobre las coordenadas
optimizadas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

COORDINATING_ELEMENTS = ("O", "N", "S")
ION_DISTANCE_A = {"O": 1.95, "N": 2.05, "S": 2.45}  # Li–X inicial; xtb lo relaja


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
        raise ValueError(f"SMILES inválido: {smiles!r}")
    return Chem.AddHs(mol)


def embed_lowest(mol: Chem.Mol, n_conformers: int = 10, seed: int = 20260924) -> tuple[Chem.Mol, int, int]:
    """Embebe `n_conformers` con ETKDG, optimiza con MMFF94 y devuelve (mol, id del
    confórmero de menor energía, número de confórmeros generados)."""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.3
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not ids:
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not ids:
        raise ValueError("RDKit no pudo embeber la molécula")
    props = AllChem.MMFFGetMoleculeProperties(mol)
    energies = []
    for cid in ids:
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            ff.Minimize(maxIts=2000)
            energies.append(ff.CalcEnergy())
        else:  # MMFF no parametriza p. ej. Si: UFF como respaldo
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            ff.Minimize(maxIts=2000)
            energies.append(ff.CalcEnergy())
    best = ids[int(np.argmin(energies))]
    return mol, best, len(ids)


def coordination_sites(mol: Chem.Mol, charges: list[float] | None = None, max_sites: int = 4) -> list[int]:
    """Índices de heteroátomos coordinantes (O, N, S con par libre), ordenados por
    carga de Mulliken más negativa si se dan cargas, si no por número atómico."""
    sites = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() in COORDINATING_ELEMENTS and a.GetTotalNumHs() == 0 and not a.GetIsAromatic()
    ]
    if charges and len(charges) == mol.GetNumAtoms():
        sites.sort(key=lambda i: charges[i])
    return sites[:max_sites]


def lone_pair_direction(struct: Structure, mol: Chem.Mol, site: int) -> np.ndarray:
    """Dirección unitaria desde el sitio, opuesta al centroide de sus vecinos."""
    neigh = [n.GetIdx() for n in mol.GetAtomWithIdx(site).GetNeighbors()]
    if not neigh:
        return np.array([1.0, 0.0, 0.0])
    v = struct.coords[site] - struct.coords[neigh].mean(axis=0)
    norm = np.linalg.norm(v)
    if norm < 1e-6:  # sitio lineal (p. ej. O de éter perfectamente simétrico): perpendicular
        a = struct.coords[neigh[0]] - struct.coords[site]
        v = np.cross(a, [0.0, 0.0, 1.0])
        if np.linalg.norm(v) < 1e-6:
            v = np.cross(a, [0.0, 1.0, 0.0])
        norm = np.linalg.norm(v)
    return v / norm


def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matriz de rotación que lleva el vector unitario `a` sobre `b` (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            return np.eye(3)
        # antiparalelos: rotar 180° alrededor de cualquier eje perpendicular
        p = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(p) < 1e-8:
            p = np.cross(a, [0.0, 1.0, 0.0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def place_ion(struct: Structure, mol: Chem.Mol, site: int, symbol: str = "Li") -> Structure:
    """Complejo molécula + ion monoatómico colocado sobre el par libre del sitio."""
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
    """Complejo molécula + cluster (p. ej. Li2S8), con el átomo `anchor` del
    cluster sobre el par libre del sitio y el resto del cluster apuntando hacia
    afuera. Si queda algún contacto por debajo de `min_separation` Å, se aleja
    a lo largo de la dirección del par libre."""
    direction = lone_pair_direction(struct, mol, site)
    d = ION_DISTANCE_A.get(struct.symbols[site], 2.0)
    target = struct.coords[site] + d * direction
    # orientar: vector ancla -> centroide del cluster alineado con `direction`
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
