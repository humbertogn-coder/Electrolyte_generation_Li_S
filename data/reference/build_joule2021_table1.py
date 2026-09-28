"""Transcription of Table 1 of Liu et al., "Electrolyte solutions design for
lithium-sulfur batteries", Joule 5, 2323-2364 (2021), doi:10.1016/j.joule.2021.06.009.

Solubility of S8 and Li2Sx in selected solvents and solutions, in mM of sulfur
atoms, at room temperature unless noted. Superscript numbers in the table are
the review's own reference numbers and are kept in `ref`. "~" in the table is
kept as `approx = True`. Cells with "n/a" or "-" produce no row.

Run this script to regenerate `joule2021_table1_solubility.csv`; edit the
ROWS list below to correct a transcription error, never the CSV by hand.

Categories follow the review: MSE = moderately solvating, SSE = sparingly
solvating, HSE = highly solvating electrolytes.
"""

from __future__ import annotations

import csv
from pathlib import Path

SPECIES = ["S8", "Li2S8", "Li2S6", "Li2S4", "Li2S3", "Li2S2", "Li2S"]

# Molecular solvents that PACE-S can represent as SMILES (ionic liquids and
# salts have no entry and get an empty solvent_smiles).
SMILES = {
    "DME": "COCCOC",
    "DOL": "C1COCO1",
    "THF": "C1CCOC1",
    "G1": "COCCOC",
    "G2": "COCCOCCOC",
    "G3": "COCCOCCOCCOC",
    "G4": "COCCOCCOCCOCCOC",
    "TTE": "FC(F)C(F)(F)OCC(F)(F)C(F)F",
    "HFE": "FC(F)C(F)(F)OCC(F)(F)C(F)F",  # HFE in the Watanabe papers is TTE
    "ACN": "CC#N",
    "DMSO": "CS(C)=O",
    "DMF": "CN(C)C=O",
    "DMA": "CN(C)C(C)=O",
    "DMI": "CN1CCN(C)C1=O",
    "TMU": "CN(C)C(=O)N(C)C",
    "SL": "O=S1(=O)CCCC1",  # sulfolane
    "tetramethylene sulfone": "O=S1(=O)CCCC1",
    "CPL": "O=C1CCCCCN1",  # caprolactam
    "acetamide": "CC(N)=O",
    "HME": None,  # abbreviation not resolved from the visible footnotes; fill in from the review
}

# (system label, category, solvent, diluent, salt, salt_conc, temperature_C, note,
#  {species: [(value, approx, ref), ...]})
ROWS = [
    # --- moderately solvating (MSE) -------------------------------------------------
    ("DME", "MSE", "DME", "", "", "", 25, "",
     {"S8": [(9.957, False, "81")], "Li2S": [(0.006, False, "81")]}),
    ("DOL", "MSE", "DOL", "", "", "", 25, "Li2S8 given as '-' in the table",
     {"Li2S": [(0.0209, False, "82")]}),
    ("THF", "MSE", "THF", "", "", "", 25, "",
     {"Li2S8": [(10000, True, "70")], "Li2S6": [(6500, True, "70")], "Li2S4": [(600, True, "70")],
      "Li2S3": [(300, True, "70")], "Li2S2": [(100, True, "70")]}),
    ("C4mpyr-Tf", "MSE", "C4mpyr-Tf (ionic liquid)", "", "", "", 25, "N-butyl-N-methylpyrrolidinium triflate",
     {"Li2S8": [(7660, False, "72")]}),
    ("DOL/DME (1:1 v/v)", "MSE", "DOL/DME", "", "", "", 25, "three literature values for Li2S8, two for Li2S6",
     {"S8": [(10, False, "72")],
      "Li2S8": [(7000, False, "83"), (5700, True, "70"), (6400, True, "84")],
      "Li2S6": [(6000, False, "83"), (1100, True, "70")],
      "Li2S4": [(2000, False, "83")]}),
    ("1 M LiTFSI in DOL/DME (1:1 v/v)", "MSE", "DOL/DME", "", "LiTFSI", "1 M", 25, "",
     {"Li2S8": [(4000, False, "85")]}),
    ("1 M LiTFSI + 0.2 M LiNO3 in DOL/DME (1:1 v/v)", "MSE", "DOL/DME", "", "LiTFSI + LiNO3", "1 M + 0.2 M", 25, "",
     {"Li2S8": [(6600, False, "44")]}),
    ("0.98 M LiTFSI in G4", "MSE", "G4", "", "LiTFSI", "0.98 M", 25, "G4 = tetraglyme (TEGDME)",
     {"S8": [(4, False, "73")], "Li2S8": [(6046, False, "73")], "Li2S4": [(34.1, False, "73")],
      "Li2S2": [(13.9, False, "73")], "Li2S": [(0.8, False, "73")]}),
    ("Li(G3)4-TFSI", "MSE", "G3", "", "LiTFSI", "1:4 Li:G3", 25, "G3 = triglyme",
     {"S8": [(3.1, True, "86")], "Li2S8": [(5889, True, "86")], "Li2S4": [(31, True, "86")],
      "Li2S2": [(13, True, "86")], "Li2S": [(1, True, "86")]}),
    ("Li(G4)4-TFSI", "MSE", "G4", "", "LiTFSI", "1:4 Li:G4", 25, "",
     {"S8": [(4, True, "86")], "Li2S8": [(6000, False, "86")], "Li2S4": [(34, True, "86")],
      "Li2S2": [(15, True, "86")], "Li2S": [(1, True, "86")]}),
    ("Li(THF)4-TFSI", "MSE", "THF", "", "LiTFSI", "1:4 Li:THF", 25, "",
     {"S8": [(4, True, "87")], "Li2S8": [(1387, False, "87")], "Li2S6": [(1250, True, "87")],
      "Li2S4": [(265, True, "87")], "Li2S2": [(25, True, "87")]}),
    ("Li(G3)1-NO3 (4.77 M)", "MSE", "G3", "", "LiNO3", "4.77 M", 25, "value reported as > 6,000",
     {"Li2S8": [(6000, True, "80")]}),
    # --- sparingly solvating (SSE) ----------------------------------------------------
    ("TTE (70 C)", "SSE", "TTE", "", "", "", 70, "",
     {"Li2S6": [(3.2, False, "88")]}),
    ("DEME-TFSI", "SSE", "DEME-TFSI (ionic liquid)", "", "", "", 25, "N,N-diethyl-N-methyl-N-(2-methoxyethyl)ammonium TFSI",
     {"S8": [(0.146, True, "72,73")], "Li2S8": [(52.7, True, "72,73")], "Li2S4": [(65.1, True, "72,73")],
      "Li2S2": [(46.5, True, "72,73")], "Li2S": [(0.5, True, "72,73")]}),
    ("C3mpyr-TFSI", "SSE", "C3mpyr-TFSI (ionic liquid)", "", "", "", 25, "",
     {"S8": [(0.146, True, "72")], "Li2S8": [(89, True, "72")], "Li2S4": [(93, True, "72")], "Li2S2": [(42, True, "72")]}),
    ("C4mpyr-TFSI", "SSE", "C4mpyr-TFSI (ionic liquid)", "", "", "", 25, "",
     {"S8": [(0.146, True, "72")], "Li2S8": [(77, True, "72")], "Li2S4": [(80, True, "72")], "Li2S2": [(20, True, "72")]}),
    ("P2225-TFSI", "SSE", "P2225-TFSI (ionic liquid)", "", "", "", 25, "triethylpentylphosphonium TFSI",
     {"S8": [(0.7, True, "72")], "Li2S8": [(39, True, "72")], "Li2S4": [(50, True, "72")], "Li2S2": [(17, True, "72")]}),
    ("C3mpip-TFSI", "SSE", "C3mpip-TFSI (ionic liquid)", "", "", "", 25, "",
     {"S8": [(0.146, True, "72")], "Li2S8": [(53, True, "72")], "Li2S4": [(49, True, "72")], "Li2S2": [(16, True, "72")]}),
    ("C4dmim-TFSI", "SSE", "C4dmim-TFSI (ionic liquid)", "", "", "", 25, "",
     {"S8": [(0.2, True, "72")], "Li2S8": [(49, True, "72")], "Li2S4": [(43, True, "72")], "Li2S2": [(26, True, "72")]}),
    ("Li(G3)1-TFSI", "SSE", "G3", "", "LiTFSI", "1:1 Li:G3 (solvate ionic liquid)", 25, "Li2S given as < 2",
     {"S8": [(0.5, True, "86")], "Li2S8": [(29, True, "86")], "Li2S6": [(40, True, "86")],
      "Li2S4": [(31, True, "86")], "Li2S2": [(15, False, "87")], "Li2S": [(2, True, "86")]}),
    ("Li(G4)1-TFSI", "SSE", "G4", "", "LiTFSI", "1:1 Li:G4 (solvate ionic liquid)", 25, "",
     {"S8": [(0.6, True, "86")], "Li2S8": [(62, True, "86")], "Li2S4": [(34, True, "86")],
      "Li2S2": [(17, True, "86")], "Li2S": [(1, True, "86")]}),
    ("Li(G4)1-TFSI/HFE (1:4 mol)", "SSE", "G4", "HFE", "LiTFSI", "1:1 Li:G4, diluted 1:4 with HFE", 25, "",
     {"S8": [(0.1, True, "86")], "Li2S8": [(10, True, "86")], "Li2S4": [(6, True, "86")],
      "Li2S2": [(9, True, "86")], "Li2S": [(1, True, "86")]}),
    ("Li(G2)4/3-TFSI", "SSE", "G2", "", "LiTFSI", "1:4/3 Li:G2", 25, "G2 = diglyme",
     {"Li2S8": [(61, False, "87")], "Li2S6": [(17, True, "87")], "Li2S4": [(17, True, "87")], "Li2S2": [(15, True, "87")]}),
    ("Li(G1)2-TFSI", "SSE", "G1", "", "LiTFSI", "1:2 Li:G1", 25, "G1 = DME",
     {"Li2S8": [(201, False, "87")], "Li2S6": [(160, True, "87")], "Li2S4": [(165, True, "87")], "Li2S2": [(17, True, "87")]}),
    ("1.5 M LiTFSI in tetramethylene sulfone/TTE (1:1 v/v)", "SSE", "tetramethylene sulfone", "TTE", "LiTFSI", "1.5 M", 25, "",
     {"Li2S8": [(264, False, "89")]}),
    ("0.64 M LiTFSI in DEME-TFSI", "SSE", "DEME-TFSI (ionic liquid)", "", "LiTFSI", "0.64 M", 25, "",
     {"S8": [(0.11, False, "73")], "Li2S8": [(6.96, False, "73")], "Li2S4": [(10.2, False, "73")],
      "Li2S2": [(5.04, False, "73")], "Li2S": [(0.5, False, "73")]}),
    ("(ACN)2-LiTFSI/TTE (1:1 v/v) (70 C)", "SSE", "ACN", "TTE", "LiTFSI", "1:2 Li:ACN, 1:1 v/v with TTE", 70, "",
     {"Li2S6": [(5.82, False, "88")]}),
    ("1 M LiTDI in DOL/DME (1:1 v/v) (30 C)", "SSE", "DOL/DME", "", "LiTDI", "1 M", 30, "",
     {"Li2S8": [(672, False, "85")]}),
    ("2 M LiTFSI in HME/DOL (9:1 v/v)", "SSE", "HME/DOL", "", "LiTFSI", "2 M", 25, "HME abbreviation to confirm in the review",
     {"Li2S8": [(400, False, "90")]}),
    ("G2:LiTFSI (0.8:1 mol) (25 C)", "SSE", "G2", "", "LiTFSI", "0.8:1 G2:Li", 25, "",
     {"Li2S6": [(12.8, False, "59")]}),
    ("G2:LiTFSI (0.8:1 mol) (55 C)", "SSE", "G2", "", "LiTFSI", "0.8:1 G2:Li", 55, "",
     {"Li2S6": [(30.5, False, "59")]}),
    ("(SL)2-LiTFSI (2.97 M)", "SSE", "SL", "", "LiTFSI", "2.97 M (1:2 Li:sulfolane)", 25, "",
     {"Li2S8": [(52, True, "91")]}),
    ("(SL)2-LiTFSI/HFE (1:4 mol)", "SSE", "SL", "HFE", "LiTFSI", "1:2 Li:SL, diluted 1:4 with HFE", 25, "",
     {"Li2S8": [(1, True, "91")]}),
    # --- highly solvating (HSE) --------------------------------------------------------
    ("TMU/DOL (1:1 v/v)", "HSE", "TMU/DOL", "", "", "", 25, "TMU = tetramethylurea",
     {"Li2S8": [(10500, False, "71")], "Li2S4": [(4000, False, "71")]}),
    ("CPL/acetamide (1:1 mol)", "HSE", "CPL/acetamide", "", "", "", 25, "Li2S3 given as '-'",
     {"S8": [(15, False, "92")], "Li2S8": [(5600, False, "92")], "Li2S6": [(4200, False, "92")],
      "Li2S4": [(2800, False, "92")], "Li2S2": [(1400, False, "92")], "Li2S": [(700, False, "92")]}),
    ("CPL/acetamide : DOL/DME (1:1 v/v)", "HSE", "CPL/acetamide + DOL/DME", "", "", "", 25, "",
     {"Li2S8": [(5000, False, "92")], "Li2S6": [(4000, False, "92")], "Li2S4": [(1600, False, "92")],
      "Li2S2": [(700, False, "92")], "Li2S": [(350, False, "92")]}),
    ("DMSO30:LiTf (0.5 M LiTf)", "HSE", "DMSO", "", "LiTf", "0.5 M", 25, "",
     {"Li2S8": [(10000, True, "70")], "Li2S6": [(8000, True, "70")], "Li2S4": [(6400, True, "70")],
      "Li2S3": [(3750, True, "70")], "Li2S2": [(2600, True, "70")]}),
    ("DMSO10:NH4NO3", "HSE", "DMSO", "", "NH4NO3", "1:10 NH4NO3:DMSO", 25, "",
     {"Li2S8": [(6000, True, "93")], "Li2S6": [(3000, True, "93")], "Li2S3": [(4700, True, "93")],
      "Li2S2": [(3600, True, "93")], "Li2S": [(1300, True, "93")]}),
    ("DMSO", "HSE", "DMSO", "", "", "", 25, "two literature values for Li2S",
     {"S8": [(3.936, False, "81")], "Li2S8": [(14250, True, "70")], "Li2S6": [(6000, True, "70")],
      "Li2S4": [(1500, True, "70")], "Li2S3": [(800, True, "70")], "Li2S2": [(300, True, "70")],
      "Li2S": [(0.02, False, "61"), (0.14, False, "82")]}),
    ("DMI", "HSE", "DMI", "", "", "", 25, "DMI = 1,3-dimethyl-2-imidazolidinone",
     {"Li2S8": [(36000, False, "84")]}),
    ("DMF", "HSE", "DMF", "", "", "", 25, "",
     {"S8": [(5.944, False, "81")], "Li2S8": [(32000, True, "84")]}),
    ("DMA", "HSE", "DMA", "", "", "", 25, "DMA = N,N-dimethylacetamide",
     {"Li2S8": [(32000, True, "84")]}),
]


def solvent_smiles(solvent: str) -> str:
    parts = [p.strip() for p in solvent.replace(" + ", "/").split("/")]
    smi = [SMILES.get(p) for p in parts]
    if any(s is None for s in smi):
        return ""
    return ".".join(smi)


def build(out: Path) -> int:
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system_id", "system", "category", "solvent", "solvent_smiles", "diluent", "salt", "salt_conc",
                    "temperature_C", "species", "solubility_mM_S", "approx", "ref", "note", "source"])
        for i, (label, cat, solv, dil, salt, conc, temp, note, data) in enumerate(ROWS, 1):
            sid = f"J21-{i:02d}"
            for sp in SPECIES:
                for value, approx, ref in data.get(sp, []):
                    w.writerow([sid, label, cat, solv, solvent_smiles(solv), dil, salt, conc, temp, sp, value,
                                approx, ref, note, "Liu et al., Joule 5, 2323 (2021), Table 1"])
                    n += 1
    return n


if __name__ == "__main__":
    here = Path(__file__).parent
    n = build(here / "joule2021_table1_solubility.csv")
    print(f"{len(ROWS)} systems, {n} solubility values written")
