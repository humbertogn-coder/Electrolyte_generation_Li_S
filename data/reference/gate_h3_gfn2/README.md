# H3 gate, first evaluation (2026-09-28)

`python -m pace_s.analysis.gate_h3` run with xtb 6.7.1 (GFN2-xTB/ALPB ether, RDKit
conformers, 4 threads, Linux x86_64) on the 27 systems of Table 1 of Joule 2021
whose solvent is molecular and that report Li2S8 solubility; 14 of them have
no salt or <= 1 M salt.

| File | Content |
|---|---|
| `gate_components_tier0.csv` | tier-0 descriptors of the 15 component molecules (cache; delete to recompute) |
| `gate_h3.csv` | one row per system: measured solubility (geometric mean of the literature values) and the aggregated descriptors (`|min`, `|mean` over components) |
| `gate_h3_correlations.csv` | Spearman of every descriptor with log10 solubility, on both sets |
| `gate_h3_loo.csv` | leave-one-out linear fits on the dilute set |
| `ptb_components.csv` | the same 15 molecules with `xtb --ptb` (PTB: property-targeted tight binding) orbital energies and dipoles next to the GFN2 ones |

Result: the central descriptor `tier0.li2s8_binding_eV` does NOT pass
(Spearman -0.51 on the dilute set, +0.08 on all systems; threshold 0.6). The
donor/polarity proxies do better in-sample (ESPmin -0.71, dipole +0.70,
HOMO +0.76, n = 14) but drop to ~0.5 in leave-one-out, i.e. with 14 points no
tier-0 descriptor is robustly above 0.6. On the full set the salt
concentration dominates (solvate ionic liquids dissolve ~100x less Li2S8 than
the dilute solvent), which a single-molecule descriptor cannot see.

Consequences, to be decided with the group (see the project notes):
1. rank solvents in tier 0 by the donor/polarity proxies (ESPmin, dipole),
   which is what the Li-S literature supports, and keep the 1:1 Li2S8 binding
   energy as one feature rather than the central one (it is biased by glyme
   chelation: G3 -2.71 eV vs DME -1.53 eV);
2. move the quantitative solubility prediction to tier 1 (MD with explicit salt);
3. enlarge the calibration set: the 69-solvent COSMO-RS set of Sci. Rep. 2023
   and ComBat, both consistent single-method datasets, unlike the compiled table.

## Does a better tight-binding method change the verdict? (2026-09-29)

`xtb --ptb` (single point on the GFN2/ALPB geometry) gives much more physical
absolute values: ether HOMO -10.0 eV (experimental ionization energy 9.4-9.7 eV)
instead of -11.0, DMSO -9.7 (exp. 9.0), positive LUMOs for all molecules, and
gaps of 12-14 eV instead of the GFN2 artifact of 5 eV for amides. Dipoles are
10-25 % high but correctly ordered (DME 2.1 D vs 1.7 exp., DMF 4.7 vs 3.8).

The correlations with Li2S8 solubility on the 14 dilute systems do NOT change:
PTB HOMO rho = +0.75 (GFN2 +0.76), PTB dipole +0.68 (GFN2 +0.70), and the
leave-one-out Spearman stays at 0.35-0.59. The rankings are the same; only the
numbers are more realistic. The gate failure is therefore not a method issue
but a data/concept issue (14 heterogeneous points; solubility depends on the
solution, not on one molecule). g-xTB (Linux-only preview) was not tested.

## What the Sci. Rep. 2023 paper and SI actually contain (2026-09-29)

No per-solvent solubility numbers. The SI lists the 69 solvents, 5
anti-solvents, 2 salts and 1 additive (names only, transcribed to
`../scirep2023_materials.csv`) and Table S1 gives each material's average
attention contribution to DME, which is an interaction weight of the model,
not a solubility. The dataset itself (100k formulations, COSMO-RS with
COSMOtherm/TURBOMOLE, eight Delta_mix G values per formulation) is held by LG
Energy Solution. Their protocol is reproducible with open tools (ORCA + an
open COSMO-RS implementation), which is the path to a consistent calibration
set of our own.
