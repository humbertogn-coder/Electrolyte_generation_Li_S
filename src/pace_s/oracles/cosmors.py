"""COSMO-RS polysulfide solubility oracle (open tools: ORCA 6.1 + openCOSMO-RS).

Why: the H3 gate showed that no single GFN2-xTB descriptor ranks the measured
Li2S8 solubility (Spearman < 0.6 on the dilute Joule 2021 systems), and the
only large solubility dataset (Sci. Rep. 2023, LG Energy Solution, COSMOtherm)
is not public. This module reproduces that protocol with open software so the
group can build its own, consistent calibration set:

    ORCA BP86/def2-TZVP(D) + CPCM  ->  .orcacosmo  ->  openCOSMO-RS 24a  ->  ln gamma_inf

The infinite-dilution activity coefficient of a polysulfide in a solvent (or
mixture) is the solubility proxy: for one solute across solvents the fusion
term of the solubility equation is constant, so log10(solubility) is expected
to decrease linearly with ln gamma_inf (expected Spearman sign -1).

Two steps, run on different machines:

1. `prepare` (laptop): writes the TAB-separated structures file for the vendored
   openCOSMO-RS conformer pipeline (`third_party/openCOSMO-RS_conformer_pipeline`)
   and a CSV mapping names to SMILES.

       python -m pace_s.oracles.cosmors prepare --set gate --out data/campaigns/cosmors/gate

2. The pipeline runs on Grace (`workflows/slurm/cosmors_orca_serial.sbatch`)
   and produces `<root>/<name>/COSMO_TZVPD/<name>_c000.orcacosmo`.

3. `evaluate` (anywhere with opencosmorspy): ln gamma_inf of the polysulfides
   in every Joule 2021 molecular system and the Spearman correlation with the
   measured solubility, i.e. the H3 gate with COSMO-RS instead of tier 0.

       python -m pace_s.oracles.cosmors evaluate --root data/campaigns/cosmors/gate/runs \
           --structures data/campaigns/cosmors/gate/structures.csv --out data/campaigns/cosmors/gate/gate_h3_cosmors.csv

Solutes are modelled as neutral Li2Sx molecules built from SMILES
(`[Li]SSSSSSSS[Li]`), the same choice as the LG paper; the DFT optimization
decides the folded geometry. Salts are not part of the mixtures, so the dilute
systems (salt <= 1 M) are the meaningful validation set. Volume ratios in the
system names are converted to mole fractions with the densities in
`DENSITY_G_ML`; other multi-component solvents use equal mole fractions.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from pace_s import DATA_DIR, REPO_ROOT
from pace_s.generate.filters import canonical, hard_filter, inchikey

log = logging.getLogger("pace_s.cosmors")

PIPELINE_DIR = REPO_ROOT / "third_party" / "openCOSMO-RS_conformer_pipeline"

SOLUTES = {
    "Li2S8": "[Li]SSSSSSSS[Li]",
    "Li2S6": "[Li]SSSSSS[Li]",
    "Li2S4": "[Li]SSSS[Li]",
    "Li2S2": "[Li]SS[Li]",
    "Li2S": "[Li]S[Li]",
    "S8": "S1SSSSSSS1",
}

# liquid densities at ~25 C (g/mL) by canonical SMILES, for v/v -> mole fraction
DENSITY_G_ML = {
    "C1COCO1": 1.060,          # DOL
    "COCCOC": 0.867,           # DME
    "CN(C)C(=O)N(C)C": 0.968,  # TMU
    "O=S1(=O)CCCC1": 1.261,    # sulfolane (30 C)
    "FC(F)C(F)(F)COC(F)(F)C(F)F": 1.530,  # TTE / HFE
    "C1CCOC1": 0.889,          # THF
    "CS(C)=O": 1.100,          # DMSO
    "CN(C)C=O": 0.944,         # DMF
    "CN(C)C(C)=O": 0.937,      # DMA
    "O=C1CCCCCN1": 1.010,      # caprolactam (melt)
    "CC(N)=O": 1.159,          # acetamide (solid)
}

_RE_RATIO = re.compile(r"\(?\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*(v/v|vol|mol)\s*\)?")


# ----------------------------------------------------------------------------- prepare

def safe_name(smiles: str, label: str | None = None) -> str:
    """Directory/file-safe molecule name: `<label>_<inchikey14>` or `mol_<inchikey14>`."""
    key = inchikey(smiles)[:14]
    if label:
        label = re.sub(r"[^A-Za-z0-9]+", "", label)[:20]
        return f"{label}_{key}"
    return f"mol_{key}"


def _rows(smiles_iter, source: str, labels: dict[str, str] | None = None) -> list[dict]:
    rows = []
    for smi in smiles_iter:
        c = canonical(smi)
        mol = Chem.MolFromSmiles(c)
        rows.append({
            "name": safe_name(c, (labels or {}).get(c)),
            "smiles": c,
            "inchikey": inchikey(c),
            "charge": Chem.GetFormalCharge(mol),
            "n_heavy": mol.GetNumHeavyAtoms(),
            "mol_weight": round(Descriptors.MolWt(mol), 2),
            "source": source,
        })
    return rows


def gate_components() -> list[str]:
    from pace_s.analysis.gate_h3 import load_systems
    systems = load_systems()
    return sorted({canonical(c) for comps in systems["components"] for c in comps})


def known_solvents(config: str = "campaign_LiS.yaml") -> pd.DataFrame:
    """Seeds + the Sci. Rep. 2023 solvents/anti-solvents that pass the hard filter of `config`."""
    from pace_s.config import load_campaign
    from pace_s.generate.seeds import load_seeds
    rows = _rows([s.smiles for s in load_seeds()], "seeds", {canonical(s.smiles): s.name for s in load_seeds()})
    sci = pd.read_csv(DATA_DIR / "reference" / "scirep2023_materials.csv")
    sci = sci[sci["role"].isin(["solvent", "anti_solvent"])]
    design_space = load_campaign(config)["design_space"]
    keep = [smi for smi in sci["smiles"] if hard_filter(smi, design_space).passes]
    rows += _rows(keep, "scirep2023")
    return pd.DataFrame(rows)


def prepare(set_name: str, solutes: list[str] | None = None) -> pd.DataFrame:
    solutes = solutes or list(SOLUTES)
    rows = _rows([SOLUTES[s] for s in solutes], "solute", {canonical(SOLUTES[s]): s for s in solutes})
    if set_name == "gate":
        rows += _rows(gate_components(), "joule2021")
    elif set_name == "known":
        rows += _rows(gate_components(), "joule2021")
        rows += known_solvents().to_dict("records")
    else:
        raise ValueError(f"unknown set {set_name!r} (gate | known)")
    df = pd.DataFrame(rows).drop_duplicates("inchikey", keep="first").reset_index(drop=True)
    return df


def write_structures(df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """`structures.inp` (TAB: name, SMILES, xyz, charge, optimize) for the pipeline and
    `structures.csv` with the mapping. Returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    inp = out_dir / "structures.inp"
    with open(inp, "w", newline="\n") as fh:
        for _, r in df.iterrows():
            fh.write(f"{r['name']}\t{r['smiles']}\t\t{int(r['charge'])}\tTrue\n")
    csv = out_dir / "structures.csv"
    df.to_csv(csv, index=False)
    return inp, csv


# ----------------------------------------------------------------------------- evaluate

def orcacosmo_path(root: Path, name: str) -> Path:
    return Path(root) / name / "COSMO_TZVPD" / f"{name}_c000.orcacosmo"


def ln_gamma_inf(solute_file: Path, solvent_files: list[Path], x_solvent: list[float],
                 temperature_K: float = 298.15) -> float:
    """ln of the infinite-dilution activity coefficient of the solute in the solvent
    mixture (openCOSMO-RS 24a, pure-component reference state)."""
    from opencosmorspy import COSMORS
    from opencosmorspy.parameterization import openCOSMORS24a

    x = np.asarray(x_solvent, dtype=float)
    x = x / x.sum()
    crs = COSMORS(par=openCOSMORS24a())
    crs.add_molecule([str(solute_file)])
    for f in solvent_files:
        crs.add_molecule([str(f)])
    crs.add_job(np.concatenate([[0.0], x]), temperature_K, refst="pure_component")
    res = crs.calculate()
    return float(res["tot"]["lng"][0][0])


def mole_fractions(system: str, components: list[str]) -> list[float]:
    """Mole fractions of the solvent components of a Joule 2021 system.
    Two components with an `a:b v/v` ratio use the densities; `a:b mol` ratios are
    used directly; anything else is equal mole fractions."""
    n = len(components)
    m = _RE_RATIO.search(system)
    if n == 2 and m:
        a, b, unit = float(m.group(1)), float(m.group(2)), m.group(3)
        if unit == "mol":
            return [a / (a + b), b / (a + b)]
        dens = [DENSITY_G_ML.get(canonical(c)) for c in components]
        if all(dens):
            mw = [Descriptors.MolWt(Chem.MolFromSmiles(c)) for c in components]
            moles = [a * dens[0] / mw[0], b * dens[1] / mw[1]]
            return [v / sum(moles) for v in moles]
    return [1.0 / n] * n


def evaluate(root: Path, structures: pd.DataFrame, species: str = "Li2S8",
             max_salt_M: float = 1.0, temperature_K: float = 298.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import spearmanr
    from pace_s.analysis.gate_h3 import load_systems

    by_key = dict(zip(structures["inchikey"], structures["name"]))
    solute_name = by_key.get(inchikey(SOLUTES[species]))
    solute_file = orcacosmo_path(root, solute_name) if solute_name else None
    if solute_file is None or not solute_file.exists():
        raise FileNotFoundError(f"no .orcacosmo for {species}: {solute_file}")

    rows = []
    for _, s in load_systems(species).iterrows():
        comps = [canonical(c) for c in s["components"]]
        files = [orcacosmo_path(root, by_key.get(inchikey(c), "")) for c in comps]
        missing = [c for c, f in zip(comps, files) if not f.exists()]
        if missing:
            log.warning("skip %s: missing %s", s["system"], missing)
            continue
        x = mole_fractions(s["system"], comps)
        lng = ln_gamma_inf(solute_file, files, x, temperature_K)
        rows.append({"system_id": s["system_id"], "system": s["system"], "category": s["category"],
                     "salt_M": s["salt_M"], "solubility_mM_S": s["solubility_mM_S"],
                     "log10_solubility": float(np.log10(s["solubility_mM_S"])),
                     "components": ".".join(comps), "x": " ".join(f"{v:.3f}" for v in x),
                     f"ln_gamma_inf_{species}": lng})
    df = pd.DataFrame(rows)
    col = f"ln_gamma_inf_{species}"
    stats = []
    for label, d in (("all_molecular", df), (f"salt<={max_salt_M}M", df[df["salt_M"] <= max_salt_M])):
        if len(d) >= 4:
            rho, p = spearmanr(d[col], d["log10_solubility"])
            stats.append({"set": label, "n": len(d), "spearman": round(float(rho), 3),
                          "p_value": round(float(p), 4), "expected_sign": -1})
    return df, pd.DataFrame(stats)


# ----------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="write structures.inp/csv for the conformer pipeline")
    p.add_argument("--set", default="gate", choices=["gate", "known"])
    p.add_argument("--solutes", nargs="*", default=None, help=f"default: {' '.join(SOLUTES)}")
    p.add_argument("--out", required=True, help="output directory")
    e = sub.add_parser("evaluate", help="H3 gate with COSMO-RS ln gamma_inf")
    e.add_argument("--root", required=True, help="directory with the pipeline outputs (<name>/COSMO_TZVPD/...)")
    e.add_argument("--structures", required=True, help="structures.csv written by prepare")
    e.add_argument("--out", required=True)
    e.add_argument("--species", default="Li2S8")
    e.add_argument("--max-salt-M", type=float, default=1.0)
    e.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "prepare":
        df = prepare(args.set, args.solutes)
        inp, csv = write_structures(df, Path(args.out))
        log.info("%d molecules (%s) -> %s, %s", len(df), df["source"].value_counts().to_dict(), inp, csv)
        return 0

    structures = pd.read_csv(args.structures)
    df, stats = evaluate(Path(args.root), structures, args.species, args.max_salt_M)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    stats.to_csv(out.with_name(out.stem + "_correlations.csv"), index=False)
    for _, r in stats.iterrows():
        verdict = "PASS" if r["spearman"] <= -args.threshold else "fail"
        log.info("gate H3/COSMO-RS [%s, n=%d]: Spearman(ln gamma_inf %s, log solubility) = %+.3f (p=%.3g) -> %s",
                 r["set"], r["n"], args.species, r["spearman"], r["p_value"], verdict)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
