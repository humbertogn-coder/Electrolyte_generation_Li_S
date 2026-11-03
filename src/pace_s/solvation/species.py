"""Species, charges, multiplicity and stoichiometry of explicit solvation clusters.

A `Species` is one chemical entity (Li+, FSI-, one solvent molecule) given by a
SMILES string; its charge, element counts and electron count come from RDKit,
never from user input, so a cluster's net charge and formula cannot drift away
from its contents. A `Composition` is a multiset of species (the stoichiometry
of one cluster); `cluster_record` turns it into a record of the
`cluster_record` contract.

Cluster protocols (`PROTOCOLS`) name the families of the plan:

    li_s1       [Li(S)]+
    li_s2       [Li(S)2]+
    li_fsi      [Li(FSI)]          (contact ion pair, no solvent)
    li_fsi_s1   [Li(FSI)(S)]

plus the coordination-saturated variants `li_sat` / `li_fsi_sat`, whose number
of solvent molecules depends on the solvent (see `n_ligands_to_saturate`).

Why the saturated variants exist (docs/adr/0001): [Li(S)2]+ gives Li+ four
donor atoms with DME but only two with a monodentate ether (THP, DBE) or a
diluent (TTE). An exchange energy built on li_s2 would then mix "how strongly
does FSI- compete" with "how many donor atoms does S have", the same kind of
bias (glyme chelation) that broke the Li2S8 affinity descriptor in Li-S.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from rdkit import Chem

LI_SMILES = "[Li+]"
FSI_SMILES = "O=S(=O)(F)[N-]S(=O)(=O)F"

ROLES = ("li_cation", "anion", "solvent", "diluent")


class SpeciesError(ValueError):
    """Invalid species or composition (bad SMILES, unknown role, empty cluster)."""


def _mol_with_hs(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SpeciesError(f"invalid SMILES: {smiles!r}")
    return Chem.AddHs(mol)


@dataclass(frozen=True)
class Species:
    """One chemical entity. `name` is a label (e.g. 'DME', 'FSI-') used in
    composition labels; `molecule_id` links solvents/diluents to the candidate
    table (PSM-xxxxxx) and is None for Li+ and FSI-."""

    name: str
    smiles: str
    role: str
    molecule_id: str | None = None
    _mol: Chem.Mol = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise SpeciesError(f"unknown role {self.role!r}; expected one of {ROLES}")
        if not self.name or any(c in self.name for c in "() "):
            raise SpeciesError(f"species name must be non-empty without spaces or parentheses: {self.name!r}")
        object.__setattr__(self, "_mol", _mol_with_hs(self.smiles))

    @property
    def mol(self) -> Chem.Mol:
        return Chem.Mol(self._mol)

    @property
    def canonical_smiles(self) -> str:
        return Chem.MolToSmiles(Chem.RemoveHs(self._mol))

    @property
    def charge(self) -> int:
        return Chem.GetFormalCharge(self._mol)

    @property
    def formula(self) -> Counter:
        return Counter(a.GetSymbol() for a in self._mol.GetAtoms())

    @property
    def n_electrons(self) -> int:
        return sum(a.GetAtomicNum() for a in self._mol.GetAtoms()) - self.charge

    @property
    def n_radical_electrons(self) -> int:
        return sum(a.GetNumRadicalElectrons() for a in self._mol.GetAtoms())

    @property
    def multiplicity(self) -> int:
        """Lowest spin consistent with the electron count (1 for every closed-shell
        species of this project; 2 for a radical)."""
        return 1 + (self.n_electrons % 2)

    def donor_atoms(self) -> list[int]:
        """Indices of atoms that can coordinate Li+ (heuristic, see `donor_atoms`)."""
        return donor_atoms(self._mol)

    @property
    def n_donor_atoms(self) -> int:
        return len(self.donor_atoms())


def li_cation() -> Species:
    return Species("Li+", LI_SMILES, "li_cation")


def fsi_anion() -> Species:
    return Species("FSI-", FSI_SMILES, "anion")


def donor_atoms(mol: Chem.Mol) -> list[int]:
    """Atoms that can donate a lone pair to Li+:

    - every neutral oxygen (ether, carbonyl, sulfonyl, phosphoryl, ester);
    - neutral sp3 nitrogen that is not bonded to a carbonyl/sulfonyl/phosphoryl
      atom (amide-like N lone pairs are delocalized), and aromatic N with no H;
    - neutral divalent sulfur (thioether); sulfoxide/sulfone S is excluded.

    Fluorine is not counted. This is a counting heuristic for choosing cluster
    sizes; the coordination actually reached is measured after optimization.
    """

    def _acyl_like(atom: Chem.Atom) -> bool:
        for nb in atom.GetNeighbors():
            for b in nb.GetBonds():
                other = b.GetOtherAtom(nb)
                if b.GetBondType() == Chem.BondType.DOUBLE and other.GetSymbol() in ("O", "S", "N"):
                    return True
        return False

    out = []
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            continue
        sym = atom.GetSymbol()
        if sym == "O":
            out.append(atom.GetIdx())
        elif sym == "N":
            if atom.GetIsAromatic():
                if atom.GetTotalNumHs() == 0 and atom.GetDegree() == 2:
                    out.append(atom.GetIdx())
            elif atom.GetHybridization() == Chem.HybridizationType.SP3 and not _acyl_like(atom):
                out.append(atom.GetIdx())
        elif sym == "S":
            if atom.GetDegree() == 2 and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()):
                out.append(atom.GetIdx())
    return out


def n_ligands_to_saturate(solvent: Species, target_donors: int = 4, max_ligands: int = 4) -> int:
    """Number of solvent molecules needed to offer Li+ about `target_donors` donor
    atoms (4 is the usual first-shell coordination of Li+ in ethers):

        DME (2 O) -> 2, THP (1 O) -> 4, triglyme (4 O) -> 1, TTE (1 O) -> 4.

    Donor atoms of one molecule beyond `target_donors` do not reduce the count
    below 1. Molecules without donor atoms raise: they cannot coordinate Li+.
    """
    n = solvent.n_donor_atoms
    if n == 0:
        raise SpeciesError(f"{solvent.name} has no donor atoms; it cannot coordinate Li+")
    return min(max_ligands, math.ceil(target_donors / min(n, target_donors)))


@dataclass(frozen=True)
class Composition:
    """Stoichiometry of one cluster or isolated species: a tuple of (Species, count)
    in a canonical order (Li+, anions, then solvents/diluents by name)."""

    parts: tuple[tuple[Species, int], ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise SpeciesError("empty composition")
        merged: dict[str, tuple[Species, int]] = {}
        for sp, n in self.parts:
            if not isinstance(n, int) or n < 1:
                raise SpeciesError(f"count of {sp.name} must be a positive integer, got {n!r}")
            if sp.name in merged:
                prev, m = merged[sp.name]
                if prev.canonical_smiles != sp.canonical_smiles:
                    raise SpeciesError(f"two different species share the name {sp.name!r}")
                merged[sp.name] = (prev, m + n)
            else:
                merged[sp.name] = (sp, n)
        order = {r: i for i, r in enumerate(ROLES)}
        parts = tuple(sorted(merged.values(), key=lambda p: (order[p[0].role], p[0].name)))
        object.__setattr__(self, "parts", parts)

    @classmethod
    def of(cls, *items: tuple[Species, int] | Species) -> "Composition":
        parts = [(it, 1) if isinstance(it, Species) else it for it in items]
        return cls(tuple(parts))

    @property
    def label(self) -> str:
        """Deterministic text key, e.g. 'Li+(1)FSI-(1)DME(1)'."""
        return "".join(f"{sp.name}({n})" for sp, n in self.parts)

    @property
    def net_charge(self) -> int:
        return sum(sp.charge * n for sp, n in self.parts)

    @property
    def formula(self) -> Counter:
        total: Counter = Counter()
        for sp, n in self.parts:
            for el, k in sp.formula.items():
                total[el] += k * n
        return total

    @property
    def formula_string(self) -> str:
        """Hill order: C, H, then the rest alphabetically (alphabetical if no C)."""
        f = self.formula
        keys = (["C", "H"] + sorted(k for k in f if k not in ("C", "H"))) if "C" in f else sorted(f)
        return "".join(f"{k}{f[k] if f[k] > 1 else ''}" for k in keys if f.get(k))

    @property
    def n_electrons(self) -> int:
        return sum(sp.n_electrons * n for sp, n in self.parts)

    @property
    def multiplicity(self) -> int:
        return 1 + (self.n_electrons % 2)

    @property
    def n_molecules(self) -> int:
        return sum(n for _, n in self.parts)

    def count(self, role: str) -> int:
        return sum(n for sp, n in self.parts if sp.role == role)

    @property
    def n_donor_atoms_offered(self) -> int:
        """Donor atoms of the neutral ligands (solvent/diluent) in the cluster."""
        return sum(sp.n_donor_atoms * n for sp, n in self.parts if sp.role in ("solvent", "diluent"))


# protocol name -> (n Li+, n FSI-, n solvent or "sat")
PROTOCOLS: dict[str, tuple[int, int, int | str]] = {
    "isolated_solvent": (0, 0, 1),
    "isolated_li": (1, 0, 0),
    "isolated_fsi": (0, 1, 0),
    "li_s1": (1, 0, 1),
    "li_s2": (1, 0, 2),
    "li_fsi": (1, 1, 0),
    "li_fsi_s1": (1, 1, 1),
    "li_sat": (1, 0, "sat"),
    "li_fsi_sat": (1, 1, "sat_minus_1"),
}


def build_composition(protocol: str, solvent: Species | None = None, target_donors: int = 4) -> Composition:
    """Composition of one protocol for one solvent. For `li_sat` the solvent count
    is `n_ligands_to_saturate`; `li_fsi_sat` has one solvent fewer (FSI- takes
    one ligand slot), never fewer than zero."""
    if protocol not in PROTOCOLS:
        raise SpeciesError(f"unknown protocol {protocol!r}; expected one of {sorted(PROTOCOLS)}")
    n_li, n_fsi, n_s = PROTOCOLS[protocol]
    if n_s in ("sat", "sat_minus_1") or (isinstance(n_s, int) and n_s > 0):
        if solvent is None:
            raise SpeciesError(f"protocol {protocol!r} needs a solvent")
        if solvent.role not in ("solvent", "diluent"):
            raise SpeciesError(f"{solvent.name} has role {solvent.role!r}, not solvent/diluent")
    if n_s == "sat":
        n_s = n_ligands_to_saturate(solvent, target_donors)
    elif n_s == "sat_minus_1":
        n_s = n_ligands_to_saturate(solvent, target_donors) - 1
    parts: list[tuple[Species, int]] = []
    if n_li:
        parts.append((li_cation(), n_li))
    if n_fsi:
        parts.append((fsi_anion(), n_fsi))
    if n_s:
        parts.append((solvent, n_s))
    return Composition(tuple(parts))


def cluster_record(
    composition: Composition,
    cluster_id: str,
    campaign_id: str,
    protocol: str,
    generation_seed: int,
    pose_index: int | None = None,
    geometry_id: str | None = None,
    status: str = "proposed",
    notes: str | None = None,
) -> dict[str, Any]:
    """Record of the `cluster_record` contract (v1.0.0). Charge, multiplicity and
    formula are derived from the composition, not passed in."""
    rec: dict[str, Any] = {
        "schema_version": "1.0.0",
        "cluster_id": cluster_id,
        "campaign_id": campaign_id,
        "protocol": protocol,
        "composition_label": composition.label,
        "species": [
            {
                "role": sp.role,
                "name": sp.name,
                "smiles": sp.canonical_smiles,
                "molecule_id": sp.molecule_id,
                "count": n,
                "charge": sp.charge,
            }
            for sp, n in composition.parts
        ],
        "net_charge": composition.net_charge,
        "multiplicity": composition.multiplicity,
        "formula": composition.formula_string,
        "n_donor_atoms_offered": composition.n_donor_atoms_offered,
        "generation_seed": generation_seed,
        "status": status,
    }
    if pose_index is not None:
        rec["pose_index"] = pose_index
    if geometry_id is not None:
        rec["geometry_id"] = geometry_id
    if notes:
        rec["notes"] = notes
    return rec


def make_id(prefix: str, n: int) -> str:
    """'PSC', 7 -> 'PSC-000007'. Prefixes: PSC cluster, PCC calculation, PSL selection."""
    if prefix not in ("PSC", "PCC", "PSL"):
        raise ValueError(f"unknown id prefix {prefix!r}")
    if not 0 <= n <= 999_999:
        raise ValueError("id number out of range 0-999999")
    return f"{prefix}-{n:06d}"


def species_from_rows(rows: Iterable[dict[str, str]], role: str = "solvent") -> list[Species]:
    """Species from rows with keys name, smiles (and optional candidate_id/role),
    e.g. data/reference/seeds.csv read with csv.DictReader."""
    out = []
    for r in rows:
        r_role = r.get("role", role)
        r_role = {"seed_solvent": "solvent", "seed_diluent": "diluent"}.get(r_role, r_role)
        out.append(Species(r["name"], r["smiles"], r_role, r.get("candidate_id") or None))
    return out
