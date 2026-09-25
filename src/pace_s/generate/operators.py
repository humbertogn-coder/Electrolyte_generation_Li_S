"""Operadores enumerativos de la etapa 1 (variación de conocidos).

Cada operador recibe un SMILES y devuelve el conjunto de hijos canónicos,
deduplicados y distintos del padre. Son deterministas y auditables: cada hijo
lleva el nombre del operador y el `candidate_id` del padre en la tabla maestra.

Convenciones comunes a todos:
- Solo se tocan carbonos sp3 no aromáticos; los operadores no crean cargas,
  radicales ni enlaces O-O / O-N.
- Un cambio por hijo. Las variaciones múltiples (p. ej. polifluoración) surgen
  al aplicar los operadores en rondas sucesivas desde `generate.run`.
- Los hijos NO pasan por el filtro duro aquí; eso lo hace `filters.hard_filter`
  en `generate.run`, para que el registro de por qué se descarta algo quede en
  la tabla.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from rdkit import Chem

Operator = Callable[[str], set[str]]

# Sustituyentes de `branch`, como SMILES de fragmento con el átomo de anclaje primero.
BRANCH_GROUPS = {
    "methyl": "C",
    "ethyl": "CC",
    "isopropyl": "C(C)C",
    "trifluoromethyl": "C(F)(F)F",
}

# Puentes de `bridge_insert`: (símbolo del átomo central, sustituyentes SMILES).
BRIDGES = {
    "CF2": ("C", ["F", "F"]),
    "SiMe2": ("Si", ["C", "C"]),
}

RING_SIZES_TO_CLOSE = (5, 6)  # `ring_close` forma anillos de 5 y 6 miembros
MAX_RING_TO_OPEN = 7


# ----------------------------------------------------------------------------
# utilidades
# ----------------------------------------------------------------------------


def _mol(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)


def _canon(mol: Chem.Mol) -> str | None:
    try:
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        # segunda pasada: garantiza que el SMILES sea re-parseable y canónico
        m2 = Chem.MolFromSmiles(smi)
        return None if m2 is None else Chem.MolToSmiles(m2)
    except Exception:  # noqa: BLE001 - RDKit lanza varios tipos
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
    """Quita un H implícito/explícito del átomo `idx`. False si no tiene."""
    a = rw.GetAtomWithIdx(idx)
    n_h = a.GetTotalNumHs()
    if n_h == 0:
        return False
    a.SetNumExplicitHs(n_h - 1)
    a.SetNoImplicit(True)
    return True


def _attach_fragment(mol: Chem.Mol, anchor_idx: int, frag_smiles: str) -> Chem.Mol | None:
    """Une el primer átomo de `frag_smiles` al átomo `anchor_idx`, consumiendo un H del ancla."""
    rw = Chem.RWMol(mol)
    if not _remove_one_h(rw, anchor_idx):
        return None
    frag = Chem.MolFromSmiles(frag_smiles)
    offset = rw.GetNumAtoms()
    combined = Chem.RWMol(Chem.CombineMols(rw.GetMol(), frag))
    combined.AddBond(anchor_idx, offset, Chem.BondType.SINGLE)
    return combined.GetMol()


def _acyclic_single_bonds(mol: Chem.Mol, symbols: set[str]) -> list[Chem.Bond]:
    """Enlaces simples, no en anillo, entre átomos cuyos símbolos están en `symbols`."""
    out = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        s = {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()}
        if s <= symbols and "C" in s:
            out.append(b)
    return out


def _insert_atom_in_bond(mol: Chem.Mol, bond: Chem.Bond, symbol: str, substituents: list[str]) -> Chem.Mol:
    """Rompe `bond` e inserta un átomo `symbol` (con sustituyentes) entre sus extremos."""
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
# operadores
# ----------------------------------------------------------------------------


def h_to_f(smiles: str) -> set[str]:
    """Sustituye un H por F en cada carbono sp3 con hidrógenos (una sustitución por hijo)."""
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
    """Inserta un -CH2- en cada enlace simple acíclico C-C o C-O."""
    mol = _mol(smiles)
    if mol is None:
        return set()
    children = [_insert_atom_in_bond(mol, b, "C", []) for b in _acyclic_single_bonds(mol, {"C", "O"})]
    return _collect(smiles, children)


def chain_contract(smiles: str) -> set[str]:
    """Elimina un -CH2- acíclico (carbono con exactamente dos vecinos pesados, sin
    sustituyentes) y une sus vecinos. No crea enlaces O-O ni O-N."""
    mol = _mol(smiles)
    if mol is None:
        return set()
    children = []
    for atom in mol.GetAtoms():
        if not _is_sp3_carbon(atom) or atom.IsInRing() or atom.GetDegree() != 2 or atom.GetTotalNumHs() != 2:
            continue
        n1, n2 = (n for n in atom.GetNeighbors())
        if {n1.GetSymbol(), n2.GetSymbol()} <= {"O", "N", "S"}:
            continue  # evitaría O-O, O-N, etc.
        if mol.GetBondBetweenAtoms(n1.GetIdx(), n2.GetIdx()) is not None:
            continue
        rw = Chem.RWMol(mol)
        rw.AddBond(n1.GetIdx(), n2.GetIdx(), Chem.BondType.SINGLE)
        rw.RemoveAtom(atom.GetIdx())
        children.append(rw.GetMol())
    return _collect(smiles, children)


def heteroatom_swap(smiles: str) -> set[str]:
    """Cambia un oxígeno de éter por S (tioéter) o por N-CH3 (amina terciaria), uno por hijo."""
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
    """Rompe un enlace simple de anillo (C-C o C-O, anillos de hasta 7) y tapa
    cada extremo con un metilo, dando el análogo acíclico."""
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
    """Forma un anillo de 5 o 6 miembros enlazando dos carbonos sp3 con H a
    distancia topológica 4 o 5 (uno por hijo)."""
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
    """Inserta un puente -CF2- o -Si(CH3)2- en cada enlace simple acíclico C-C o C-O.
    (El puente -CH2- es `chain_extend`.)"""
    mol = _mol(smiles)
    if mol is None:
        return set()
    children = []
    for b in _acyclic_single_bonds(mol, {"C", "O"}):
        for symbol, subs in BRIDGES.values():
            children.append(_insert_atom_in_bond(mol, b, symbol, subs))
    return _collect(smiles, children)


def branch(smiles: str) -> set[str]:
    """Añade un sustituyente (metilo, etilo, isopropilo, CF3) a cada carbono sp3 con H."""
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
    """Aplica una lista de operadores por nombre; devuelve {operador: hijos}."""
    return {name: OPERATORS[name](smiles) for name in operators}
