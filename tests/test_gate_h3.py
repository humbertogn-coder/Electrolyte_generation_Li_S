"""H3 gate machinery without xtb: table loading, aggregation, correlations, LOO."""

import numpy as np
import pandas as pd

from pace_s import DATA_DIR
from pace_s.analysis import gate_h3 as g

RESULTS = DATA_DIR / "reference" / "gate_h3_gfn2"


def test_salt_molarity():
    assert g.salt_molarity("") == 0.0
    assert g.salt_molarity(float("nan")) == 0.0
    assert g.salt_molarity("0.98 M") == 0.98
    assert g.salt_molarity("1 M + 0.2 M") == 1.0
    assert g.salt_molarity("1:1 Li:G4 (solvate ionic liquid)") == 3.0


def test_load_systems():
    s = g.load_systems("Li2S8")
    assert len(s) == 27
    assert (s["n_values"] >= 1).all()
    dol_dme = s[s["system"] == "DOL/DME (1:1 v/v)"].iloc[0]
    assert dol_dme["n_values"] == 3
    assert abs(dol_dme["solubility_mM_S"] - (7000 * 5700 * 6400) ** (1 / 3)) < 1
    hfe = s[s["system"].str.contains("HFE")].iloc[0]
    assert g.DILUENT_SMILES["HFE"] in hfe["components"]
    assert (s["salt_M"] <= 1.0).sum() == 14


def test_aggregate_and_correlations_on_stored_results():
    comp = pd.read_csv(RESULTS / "gate_components_tier0.csv").set_index("canonical_smiles")
    agg = g.aggregate(g.load_systems(), comp)
    assert len(agg) == 27
    assert "tier0.li2s8_binding_eV|min" in agg
    corr = g.correlations(agg[agg["salt_M"] <= 1.0], "dilute")
    row = corr[(corr["descriptor"] == "tier0.li2s8_binding_eV") & (corr["agg"] == "mean")].iloc[0]
    assert row["n"] == 14
    assert -0.6 < row["spearman"] < -0.4  # the recorded first evaluation: the gate fails
    esp = corr[(corr["descriptor"] == "tier0.esp_min_kcal_mol") & (corr["agg"] == "mean")].iloc[0]
    assert esp["spearman"] < -0.6


def test_loo_fit_recovers_a_linear_relation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    df = pd.DataFrame({"a": x, "solubility_mM_S": 10 ** (2 - 1.5 * x + rng.normal(scale=0.05, size=20))})
    r = g.loo_fit(df, ["a"])
    assert r["n"] == 20 and r["loo_spearman"] > 0.95 and r["loo_rmse_log10"] < 0.1
