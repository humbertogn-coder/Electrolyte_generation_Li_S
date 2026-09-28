"""Free 2D RDKit descriptors (part of tier 0)."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

from pace_s.generate.filters import fluorination_fraction, ratio_c_o


def rdkit_descriptors(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return {
        "mol_weight": Descriptors.MolWt(mol),
        "tpsa_A2": rdMolDescriptors.CalcTPSA(mol),
        "logp": Crippen.MolLogP(mol),
        "frac_fluorination": fluorination_fraction(mol),
        "ratio_C_O": ratio_c_o(mol),
    }
