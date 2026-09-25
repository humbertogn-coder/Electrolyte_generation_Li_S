"""Operadores enumerativos de la etapa 1 (variación de conocidos).

Cada operador recibe un SMILES canónico y devuelve el conjunto de hijos
canónicos, deduplicados. Son deterministas y auditables: cada hijo lleva el
nombre del operador y el `candidate_id` del padre en la tabla maestra.

Estado: solo `h_to_f` está implementado, como referencia de estilo. Los demás
se implementan en la semana 1 (ver cronograma), cada uno con su test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from rdkit import Chem

Operator = Callable[[str], set[str]]


def _canon(mol: Chem.Mol) -> str | None:
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - RDKit lanza varios tipos
        return None
    return Chem.MolToSmiles(mol)


def h_to_f(smiles: str) -> set[str]:
    """Sustituye un H por F en cada carbono sp3 con hidrógenos (una sustitución por hijo)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    children: set[str] = set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or atom.GetTotalNumHs() == 0 or atom.GetIsAromatic():
            continue
        rw = Chem.RWMol(mol)
        idx = rw.AddAtom(Chem.Atom("F"))
        rw.AddBond(atom.GetIdx(), idx, Chem.BondType.SINGLE)
        a = rw.GetAtomWithIdx(atom.GetIdx())
        a.SetNumExplicitHs(max(atom.GetTotalNumHs() - 1, 0))
        a.SetNoImplicit(True)
        child = _canon(rw.GetMol())
        if child and child != smiles:
            children.add(child)
    return children


def _not_implemented(name: str) -> Operator:
    def op(smiles: str) -> set[str]:
        raise NotImplementedError(f"Operador {name!r} pendiente (semana 1).")

    op.__name__ = name
    return op


OPERATORS: dict[str, Operator] = {
    "h_to_f": h_to_f,
    "chain_extend": _not_implemented("chain_extend"),
    "chain_contract": _not_implemented("chain_contract"),
    "heteroatom_swap": _not_implemented("heteroatom_swap"),
    "ring_open": _not_implemented("ring_open"),
    "ring_close": _not_implemented("ring_close"),
    "bridge_insert": _not_implemented("bridge_insert"),
    "branch": _not_implemented("branch"),
}


def apply(operators: Iterable[str], smiles: str) -> dict[str, set[str]]:
    """Aplica una lista de operadores por nombre; devuelve {operador: hijos}."""
    return {name: OPERATORS[name](smiles) for name in operators}
