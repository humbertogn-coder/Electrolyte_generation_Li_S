"""Stage-1 enumerative operators (variation of known molecules).

Each operator takes a SMILES and returns the set of canonical children,
deduplicated and different from the parent. They are deterministic and
auditable: every child carries the operator name and the parent's
`candidate_id` in the master table.

Conventions shared by all operators:
- Only non-aromatic sp3 carbons are touched; operators never create charges,
  radicals or O-O / O-N bonds.
- One change per child. Multiple variations (e.g. polyfluorination) emerge by
  applying the operators over successive rounds from `generate.run`.
- Children are NOT passed through the hard filter here; `filters.hard_filter`
  does that in `generate.run`, so the reason a molecule is discarded stays in
  the table.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from rdkit import Chem

Operator = Callable[[str], set[str]]

# `branch` substituents, as fragment SMILES with the anchor atom first.
BRANCH_GROUPS = {
    "methyl": "C",
    "ethyl": "CC",
    "isopropyl": "C(C)C",
    "trifluoromethyl": "C(F)(F)F",
}

# `bridge_insert` bridges: (central atom symbol, substituent SMILES).
BRIDGES = {
    "CF2": ("C", ["F", "F"]),
    "SiMe2": ("Si", ["C", "C"]),
}

RING_SIZES_TO_CLOSE = (5, 6)  # `ring_close` forms 5- and 6-membered rings
MAX_RING_TO_OPEN = 7


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _mol(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)


def _canon(mol: Chem.Mol) -> str | None:
    try:
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        # second pass: guarantees the SMILES is re-parseable and canonical
        m2 = Chem.MolFromSmiles(smi)
        return None if m2 is None else Chem.MolToSmiles(m2)
    except Exception:  # noqa: BLE001 - RDKit raises several types
        return None


def _collect(parent: str, candidates: Iterable[Chem.Mol]) -> set[str]:
    parent_canon = _canon(Chem.Mol(_mol(parent))) if _mol(parent) else None
    out: set[str] = set()
    for m in candidates:
        smi = _canon(m)
        if smi and smi != parent_canon:
            out.add(smi)
    return out


def _is_sp3_carbon(atom: Chem.Atom) -> bool:
    return (
        atom.GetSymbol() == "C"
        and not atom.GetIsAromatic()
        and atom.GetHybridization() == Chem.HybridizationType.SP3
    )


def _remove_one_h(rw: Chem.RWMol, idx: int) -> bool:
    """Remove one implicit/explicit H from atom `idx`. False if it has none."""
    a = rw.GetAtomWithIdx(idx)
    n_h = a.GetTotalNumHs()
    if n_h == 0:
        return False
    a.SetNumExplicitHs(n_h - 1)
    a.SetNoImplicit(True)
    return True


def _attach_fragment(mol: Chem.Mol, anchor_idx: int, frag_smiles: str) -> Chem.Mol | None:
    """Bond the first atom of `frag_smiles` to atom `anchor_idx`, consuming one H of the anchor."""
    rw = Chem.RWMol(mol)
    if not _remove_one_h(rw, anchor_idx):
        return None
    frag = Chem.MolFromSmiles(frag_smiles)
    offset = rw.GetNumAtoms()
    combined = Chem.RWMol(Chem.CombineMols(rw.GetMol(), frag))
    combined.AddBond(anchor_idx, offset, Chem.BondType.SINGLE)
    return combined.GetMol()


def _acyclic_single_bonds(mol: Chem.Mol, symbols: set[str]) -> list[Chem.Bond]:
    """Single, non-ring bonds between atoms whose symbols are in `symbols`."""
    out = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        s = {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()}
        if s <= symbols and "C" in s:
            out.append(b)
    return out


def _insert_atom_in_bond(mol: Chem.Mol, bond: Chem.Bond, symbol: str, substituents: list[str]) -> Chem.Mol:
    """Break `bond` and insert an atom `symbol` (with substituents) between its ends."""
    rw = Chem.RWMol(mol)
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    rw.RemoveBond(i, j)
    k = rw.AddAtom(Chem.Atom(symbol))
    rw.AddBond(i, k, Chem.BondType.SINGLE)
    rw.AddBond(k, j, Chem.BondType.SINGLE)
    for sub in substituents:
        s = rw.AddAtom(Chem.Atom(sub))
        rw.AddBond(k, s, Chem.BondType.SINGLE)
    return rw.GetMol()


# ----------------------------------------------------------------------------
# operators
# ----------------------------------------------------------------------------


def h_to_f(smiles: str) -> set[str]:
    """Replace one H by F on every sp3 carbon bearing hydrogens (one substitution per child)."""
    mol = _mol(smiles)
    if mol is None:
        return set()
    children = []
    for atom in mol.GetAtoms():
        if not _is_sp3_carbon(atom) or atom.GetTotalNumHs() == 0:
            continue
        rw = Chem.RWMol(mol)
        _remove_one_h(rw, atom.GetIdx())
        f = rw.AddAtom(Chem.Atom("F"))
        rw.AddBond(atom.GetIdx(), f, Chem.BondType.SINGLE)
        children.append(rw.GetMol())
    return _collect(smiles, children)


def chain_extend(smiles: str) -> set[str]:
    """Insert a -CH2- into every acyclic single C-C or C-O bond."""
    mol = _mol(smiles)
    if mol is None:
        return set()
    children = [_insert_atom_in_bond(mol, b, "C", []) for b in _acyclic_single_bonds(mol, {"C", "O"})]
    return _collect(smiles, children)


def chain_contract(smiles: str) -> set[str]:
    """Remove an acyclic -CH2- (carbon with exactly two heavy neighbors and no
    substituents) and bond its neighbors. Never creates O-O or O-N bonds."""
    mol = _mol(smiles)
    if mol is None:
        return set()
    children = []
    for atom in mol.GetAtoms():
        if not _is_sp3_carbon(atom) or atom.IsInRing() or atom.GetDegree() != 2 or atom.GetTotalNumHs() != 2:
            continue
        n1, n2 = (n for n in atom.GetNeighbors())
        if {n1.GetSymbol(), n2.GetSymbol()} <= {"O", "N", "S"}:
            continue  # would create O-O, O-N, etc.
        if mol.GetBondBetweenAtoms(n1.GetIdx(), n2.GetIdx()) is not None:
            continue
        rw = Chem.RWMol(mol)
        rw.AddBond(n1.GetIdx(), n2.GetIdx(), Chem.BondType.SINGLE)
        rw.RemoveAtom(atom.GetIdx())
        children.append(rw.GetMol())
    return _collect(smiles, children)


def heteroatom_swap(smiles: str) -> set[str]:
    """Swap one ether oxygen for S (thioether) or N-CH3 (tertiary amine), one per child."""
    mol = _mol(smiles)
    if mol is None:
        return set()
    children = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "O" or atom.GetDegree() != 2 or atom.GetTotalNumHs() != 0:
            continue
        # O -> S
        rw = Chem.RWMol(mol)
        rw.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(16)
        children.append(rw.GetMol())
        # O -> N(CH3)
        rw = Chem.RWMol(mol)
        a = rw.GetAtomWithIdx(atom.GetIdx())
        a.SetAtomicNum(7)
        me = rw.AddAtom(Chem.Atom("C"))
        rw.AddBond(atom.GetIdx(), me, Chem.BondType.SINGLE)
        children.append(rw.GetMol())
    return _collect(smiles, children)


def ring_open(smiles: str) -> set[str]:
    """Break one single ring bond (C-C or C-O, rings up to 7) and cap each end
    with a methyl, giving the acyclic analogue."""
    mol = _mol(smiles)
    if mol is None or mol.GetRingInfo().NumRings() == 0:
        return set()
    ring_info = mol.GetRingInfo()
    children = []
    for b in mol.GetBonds():
        if not b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        if ring_info.MinBondRingSize(b.GetIdx()) > MAX_RING_TO_OPEN:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetIsAromatic() or a2.GetIsAromatic():
            continue
        if {a1.GetSymbol(), a2.GetSymbol()} - {"C", "O"}:
            continue
        rw = Chem.RWMol(mol)
        rw.RemoveBond(a1.GetIdx(), a2.GetIdx())
        for idx in (a1.GetIdx(), a2.GetIdx()):
            me = rw.AddAtom(Chem.Atom("C"))
            rw.AddBond(idx, me, Chem.BondType.SINGLE)
        children.append(rw.GetMol())
    return _collect(smiles, children)


def ring_close(smiles: str) -> set[str]:
    """Form a 5- or 6-membered ring by bonding two H-bearing sp3 carbons at
    topological distance 4 or 5 (one per child)."""
    mol = _mol(smiles)
    if mol is None:
        return set()
    dm = Chem.GetDistanceMatrix(mol)
    children = []
    n = mol.GetNumAtoms()
    for i in range(n):
        ai = mol.GetAtomWithIdx(i)
        if not _is_sp3_carbon(ai) or ai.GetTotalNumHs() == 0:
            continue
        for j in range(i + 1, n):
            aj = mol.GetAtomWithIdx(j)
            if not _is_sp3_carbon(aj) or aj.GetTotalNumHs() == 0:
                continue
            if int(dm[i][j]) + 1 not in RING_SIZES_TO_CLOSE:
                continue
            rw = Chem.RWMol(mol)
            if not (_remove_one_h(rw, i) and _remove_one_h(rw, j)):
                continue
            rw.AddBond(i, j, Chem.BondType.SINGLE)
            children.append(rw.GetMol())
    return _collect(smiles, children)


def bridge_insert(smiles: str) -> set[str]:
    """Insert a -CF2- or -Si(CH3)2- bridge into every acyclic single C-C or C-O bond.
    (The -CH2- bridge is `chain_extend`.)"""
    mol = _mol(smiles)
    if mol is None:
        return set()
    children = []
    for b in _acyclic_single_bonds(mol, {"C", "O"}):
        for symbol, subs in BRIDGES.values():
            children.append(_insert_atom_in_bond(mol, b, symbol, subs))
    return _collect(smiles, children)


def branch(smiles: str) -> set[str]:
    """Add a substituent (methyl, ethyl, isopropyl, CF3) to every H-bearing sp3 carbon."""
    mol = _mol(smiles)
    if mol is None:
        return set()
    children = []
    for atom in mol.GetAtoms():
        if not _is_sp3_carbon(atom) or atom.GetTotalNumHs() == 0:
            continue
        for frag in BRANCH_GROUPS.values():
            child = _attach_fragment(mol, atom.GetIdx(), frag)
            if child is not None:
                children.append(child)
    return _collect(smiles, children)


OPERATORS: dict[str, Operator] = {
    "h_to_f": h_to_f,
    "chain_extend": chain_extend,
    "chain_contract": chain_contract,
    "heteroatom_swap": heteroatom_swap,
    "ring_open": ring_open,
    "ring_close": ring_close,
    "bridge_insert": bridge_insert,
    "branch": branch,
}


def apply(operators: Iterable[str], smiles: str) -> dict[str, set[str]]:
    """Apply a list of operators by name; returns {operator: children}."""
    return {name: OPERATORS[name](smiles) for name in operators}
