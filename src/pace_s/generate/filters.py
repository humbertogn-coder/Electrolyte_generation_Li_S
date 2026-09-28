"""Hard constraints of the design space (proposal sections 4 and 6).

Applied before tier 0. Thresholds live under `design_space` in the campaign
YAML; only the logic is here. The synthesizability filter (SAScore, RAScore)
and the melting/boiling point estimates live in separate modules because they
depend on external models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors


@dataclass(frozen=True)
class FilterResult:
    passes: bool
    reason: str | None = None


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToInchiKey(mol)


def longest_perfluoroalkyl_chain(mol: Chem.Mol) -> int:
    """Maximum length of a contiguous chain of -CF2- / -CF3 carbons (PFAS criterion)."""
    pf = set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        n_f = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "F")
        n_h = atom.GetTotalNumHs()
        if n_f >= 2 and n_h == 0:
            pf.add(atom.GetIdx())
    best = 0
    seen: set[int] = set()
    for start in pf:
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            i = stack.pop()
            if i in comp:
                continue
            comp.add(i)
            for n in mol.GetAtomWithIdx(i).GetNeighbors():
                if n.GetIdx() in pf and n.GetIdx() not in comp:
                    stack.append(n.GetIdx())
        seen |= comp
        best = max(best, len(comp))
    return best


def fluorination_fraction(mol: Chem.Mol) -> float:
    n_f = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "F")
    n_h = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    return 0.0 if n_f + n_h == 0 else n_f / (n_f + n_h)


def ratio_c_o(mol: Chem.Mol) -> float:
    n_c = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    n_o = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
    return float("inf") if n_o == 0 else n_c / n_o


def hard_filter(smiles: str, design_space: dict[str, Any]) -> FilterResult:
    """Apply the hard constraints of `design_space` (from the campaign YAML)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return FilterResult(False, "invalid_smiles")

    allowed = set(design_space["allowed_elements"])
    bad = sorted({a.GetSymbol() for a in mol.GetAtoms()} - allowed)
    if bad:
        return FilterResult(False, f"element_not_allowed:{','.join(bad)}")

    mw = Descriptors.MolWt(mol)
    lo, hi = design_space["mol_weight_g_mol"]["min"], design_space["mol_weight_g_mol"]["max"]
    if not lo <= mw <= hi:
        return FilterResult(False, f"mol_weight_out_of_range:{mw:.1f}")

    if Chem.GetFormalCharge(mol) != 0:
        return FilterResult(False, "charged")

    if Descriptors.NumRadicalElectrons(mol) > 0:
        return FilterResult(False, "radical")

    for smarts in design_space.get("forbidden_smarts", []):
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f"Invalid SMARTS in the configuration: {smarts!r}")
        if mol.HasSubstructMatch(patt):
            return FilterResult(False, f"forbidden_group:{smarts}")

    max_pf = design_space.get("max_perfluoroalkyl_chain")
    if max_pf is not None and longest_perfluoroalkyl_chain(mol) > max_pf:
        return FilterResult(False, "perfluoroalkyl_chain_too_long")

    return FilterResult(True)
