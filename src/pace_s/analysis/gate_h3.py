"""H3 gate: does tier 0 reproduce the measured polysulfide solubility ranking?

    python -m pace_s.analysis.gate_h3 --config workflows/configs/campaign_LiS.yaml \
        --workdir data/campaigns/gate/tier0 --out data/campaigns/gate/gate_h3.csv --threads 4

Takes the systems of Table 1 of Joule 2021 (`data/reference/joule2021_table1_solubility.csv`)
whose solvent is a molecule or a mixture of molecules, computes the tier-0
descriptors of every component with xtb, aggregates them per system, and
reports the Spearman rank correlation between each descriptor and the
measured Li2S8 solubility (mM of S atoms). The proposal's gate is
Spearman > 0.6 for the central descriptor, `tier0.li2s8_binding_eV`.

What a single-molecule descriptor can and cannot explain: solubility in the
table depends on the solvent AND on the salt concentration (a solvate ionic
liquid like Li(G4)1-TFSI dissolves ~100x less Li2S8 than dilute G4). The gate
is therefore evaluated on two sets: all molecular systems, and only those with
no salt or at most `--max-salt-M` (default 1.0 M), where the solvent identity
dominates. Mixtures are aggregated with the most strongly binding component
(`agg=min`) and with the mean (`agg=mean`); both are reported.

Systems with several literature values use their geometric mean (values span
orders of magnitude). Ionic-liquid solvents have no SMILES and are skipped.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pace_s import DATA_DIR
from pace_s.config import load_campaign
from pace_s.generate.filters import canonical

log = logging.getLogger("pace_s.gate_h3")

TABLE = DATA_DIR / "reference" / "joule2021_table1_solubility.csv"
DILUENT_SMILES = {"TTE": "FC(F)C(F)(F)OCC(F)(F)C(F)F", "HFE": "FC(F)C(F)(F)OCC(F)(F)C(F)F"}

DESCRIPTORS = [
    "tier0.li2s8_binding_eV", "tier0.li_binding_eV", "tier0.esp_min_kcal_mol", "tier0.esp_max_kcal_mol",
    "tier0.gap_eV", "tier0.homo_eV", "tier0.lumo_eV", "tier0.dipole_D", "tier0.solv_energy_kcal_mol",
    "tier0.tpsa_A2", "tier0.logp", "tier0.ratio_C_O", "tier0.frac_fluorination",
]
# expected sign of the correlation with solubility (more negative binding -> more soluble, etc.)
EXPECTED_SIGN = {
    "tier0.li2s8_binding_eV": -1, "tier0.li_binding_eV": -1, "tier0.esp_min_kcal_mol": -1,
    "tier0.esp_max_kcal_mol": -1, "tier0.dipole_D": +1, "tier0.solv_energy_kcal_mol": -1,
}

_RE_MOLAR = re.compile(r"^\s*([0-9.]+)\s*M\b")


def salt_molarity(conc: str | float) -> float:
    """Rough molarity from the `salt_conc` text: '' -> 0, '0.98 M' -> 0.98,
    solvate ratios like '1:1 Li:G4' -> 3.0 (treated as concentrated)."""
    if conc is None or (isinstance(conc, float) and np.isnan(conc)) or str(conc).strip() == "":
        return 0.0
    m = _RE_MOLAR.match(str(conc))
    if m:
        return float(m.group(1))
    return 3.0


def load_systems(species: str = "Li2S8") -> pd.DataFrame:
    df = pd.read_csv(TABLE)
    df = df[(df["species"] == species) & df["solvent_smiles"].notna()].copy()
    g = df.groupby(["system_id", "system", "category", "solvent", "solvent_smiles"], as_index=False).agg(
        solubility_mM_S=("solubility_mM_S", lambda v: float(np.exp(np.log(v).mean()))),
        n_values=("solubility_mM_S", "size"),
        temperature_C=("temperature_C", "first"),
        salt=("salt", "first"),
        salt_conc=("salt_conc", "first"),
    )
    g["salt_M"] = g["salt_conc"].map(salt_molarity)
    diluent = df.groupby("system_id")["diluent"].first()
    g["diluent"] = g["system_id"].map(diluent).fillna("")
    g["components"] = [
        smi.split(".") + ([DILUENT_SMILES[d]] if d in DILUENT_SMILES else [])
        for smi, d in zip(g["solvent_smiles"], g["diluent"])
    ]
    return g


def loo_fit(agg: pd.DataFrame, columns: list[str]) -> dict:
    """Leave-one-out linear fit of log10(solubility) on `columns`; returns the
    Spearman correlation of the held-out predictions with the measurements.
    With n ~ 14 this is the only honest way to compare descriptor combinations."""
    d = agg.dropna(subset=columns)
    y = np.log10(d["solubility_mM_S"].to_numpy())
    X = np.column_stack([d[c].to_numpy(dtype=float) for c in columns] + [np.ones(len(d))])
    preds = np.empty(len(d))
    for i in range(len(d)):
        mask = np.arange(len(d)) != i
        coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        preds[i] = X[i] @ coef
    rho, p = spearmanr(preds, y)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    return {"columns": " + ".join(columns), "n": int(len(d)), "loo_spearman": round(float(rho), 3),
            "p_value": round(float(p), 4), "loo_rmse_log10": round(rmse, 3)}


def component_descriptors(systems: pd.DataFrame, workdir: Path, cfg: dict, threads: int | None,
                          cache: Path | None = None) -> pd.DataFrame:
    """Tier-0 descriptors of every unique component molecule (cached in `cache` CSV)."""
    from pace_s.descriptors.run_tier0 import Tier0Settings, run_block

    smiles = sorted({canonical(s) for comps in systems["components"] for s in comps})
    done = pd.DataFrame()
    if cache and cache.exists():
        done = pd.read_csv(cache)
        smiles = [s for s in smiles if s not in set(done["canonical_smiles"])]
    if smiles:
        st = Tier0Settings(cfg, threads=threads)
        new = run_block([(f"GATE-{i:03d}", s) for i, s in enumerate(smiles)], workdir, st)
        done = pd.concat([done, new], ignore_index=True)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            done.to_csv(cache, index=False)
    return done.set_index("canonical_smiles")


def aggregate(systems: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in systems.iterrows():
        parts = [comp.loc[canonical(c)] for c in s["components"] if canonical(c) in comp.index]
        if not parts or any(pd.notna(p.get("tier0.error")) for p in parts):
            continue
        row = s.to_dict()
        for d in DESCRIPTORS:
            vals = [p[d] for p in parts if pd.notna(p[d])]
            if not vals:
                continue
            row[f"{d}|min"] = float(min(vals))
            row[f"{d}|mean"] = float(np.mean(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def correlations(agg: pd.DataFrame, label: str) -> pd.DataFrame:
    out = []
    y = np.log10(agg["solubility_mM_S"])
    for d in DESCRIPTORS:
        for how in ("min", "mean"):
            col = f"{d}|{how}"
            if col not in agg or agg[col].notna().sum() < 4:
                continue
            x = agg[col]
            rho, p = spearmanr(x, y, nan_policy="omit")
            out.append({"set": label, "descriptor": d, "agg": how, "n": int(x.notna().sum()),
                        "spearman": round(float(rho), 3), "p_value": round(float(p), 4),
                        "expected_sign": EXPECTED_SIGN.get(d, 0)})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True, help="CSV with one row per system and the aggregated descriptors")
    ap.add_argument("--species", default="Li2S8")
    ap.add_argument("--max-salt-M", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None, help="default: calibration.gate.threshold")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_campaign(args.config)
    threshold = args.threshold if args.threshold is not None else float(cfg["calibration"]["gate"]["threshold"])
    systems = load_systems(args.species)
    log.info("%d molecular systems with %s data in the Joule 2021 table", len(systems), args.species)

    out = Path(args.out)
    comp = component_descriptors(systems, Path(args.workdir), cfg, args.threads,
                                 cache=out.with_name("gate_components_tier0.csv"))
    agg = aggregate(systems, comp)
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.drop(columns=["components"]).to_csv(out, index=False)

    dilute = agg[agg["salt_M"] <= args.max_salt_M]
    corr = pd.concat([correlations(agg, "all_molecular"), correlations(dilute, f"salt<={args.max_salt_M}M")],
                     ignore_index=True)
    corr.to_csv(out.with_name(out.stem + "_correlations.csv"), index=False)

    central = corr[(corr["descriptor"] == "tier0.li2s8_binding_eV")]
    log.info("systems: %d molecular, %d with salt <= %.1f M", len(agg), len(dilute), args.max_salt_M)
    for _, r in central.iterrows():
        verdict = "PASS" if abs(r["spearman"]) >= threshold and np.sign(r["spearman"]) == r["expected_sign"] else "fail"
        log.info("gate H3 [%s, agg=%s, n=%d]: Spearman(li2s8_binding, log solubility) = %+.3f (p=%.3g) -> %s (threshold %.2f, expected sign %+d)",
                 r["set"], r["agg"], r["n"], r["spearman"], r["p_value"], verdict, threshold, r["expected_sign"])
    best = corr.reindex(corr["spearman"].abs().sort_values(ascending=False).index).head(8)
    log.info("strongest single descriptors:\n%s", best.to_string(index=False))

    # leave-one-out fits on the dilute set: which small combination would pass?
    combos = [
        ["tier0.li2s8_binding_eV|min"],
        ["tier0.esp_min_kcal_mol|mean"],
        ["tier0.esp_min_kcal_mol|mean", "tier0.dipole_D|mean"],
        ["tier0.esp_min_kcal_mol|mean", "tier0.ratio_C_O|mean"],
        ["tier0.esp_min_kcal_mol|mean", "tier0.li2s8_binding_eV|min"],
        ["tier0.esp_min_kcal_mol|mean", "tier0.dipole_D|mean", "tier0.li2s8_binding_eV|min"],
    ]
    loo = pd.DataFrame([loo_fit(dilute, c) for c in combos if all(col in dilute for col in c)])
    loo.to_csv(out.with_name(out.stem + "_loo.csv"), index=False)
    log.info("leave-one-out linear fits on the salt<=%.1fM set:\n%s", args.max_salt_M, loo.to_string(index=False))
    log.info("wrote %s and %s", out, out.with_name(out.stem + "_correlations.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
