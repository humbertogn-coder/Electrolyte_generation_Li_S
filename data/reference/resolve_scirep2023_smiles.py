"""Resolve the IUPAC names of `scirep2023_materials.csv` to SMILES with OPSIN
(via py2opsin, which bundles the OPSIN jar; needs Java) and canonicalize them
with RDKit. Writes the `smiles` and `inchikey` columns in place.

    pip install py2opsin
    python data/reference/resolve_scirep2023_smiles.py

Names OPSIN cannot parse are left empty and listed at the end; fix them by
hand in the CSV (column `smiles_manual`), which takes precedence when present.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem

HERE = Path(__file__).parent
CSV = HERE / "scirep2023_materials.csv"


def main() -> None:
    from py2opsin import py2opsin

    df = pd.read_csv(CSV)
    if "smiles_manual" not in df.columns:
        df["smiles_manual"] = ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = py2opsin(df["name"].tolist(), output_format="SMILES")
    smiles, keys, failed = [], [], []
    for name, opsin, manual in zip(df["name"], raw, df["smiles_manual"].fillna("")):
        smi = manual.strip() or (opsin or "")
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            smiles.append("")
            keys.append("")
            failed.append(name)
        else:
            smiles.append(Chem.MolToSmiles(mol))
            keys.append(Chem.MolToInchiKey(mol))
    df["smiles"] = smiles
    df["inchikey"] = keys
    cols = ["name", "role", "smiles", "inchikey", "attention_to_DME", "note", "smiles_manual"]
    df[cols].to_csv(CSV, index=False)
    print(f"{len(df) - len(failed)}/{len(df)} resolved")
    for n in failed:
        print("  unresolved:", n)


if __name__ == "__main__":
    main()
