# data/reference

Reference and calibration data. Everything here is small, versionable and sourced.

| File | Content | Status |
|---|---|---|
| `seeds.csv` | Coordinating and diluent seeds (M0), each with the paper it comes from (`reference`, `doi`). | F5DEE and OFE SMILES flagged for confirmation against the papers' SI |
| `salts.csv` | LiFSI and LiTFSI. | ready |
| `tier0_reference_molecules.csv` | Five molecules with the expected tier-0 ordering (H1 success criterion). | numeric DN with source still to fill |
| `li2s8_gfn2_alpb_ether.xyz` | Li₂S₈ reference geometry (GFN2-xTB/ALPB) for the tier-0 affinity. Not a verified global minimum. | ready |
| `tier0_reference_results_gfn2.csv` | Tier-0 results for the 6 reference molecules (xtb 6.7.1, Linux). Regression set for `tests/test_tier0.py`. | ready |
| `joule2021_table1_solubility.csv` | Table 1 of *Joule* 2021: S₈ and Li₂Sₓ solubility in mM for 40+ systems. **H3 gate dataset.** | **to digitize (week 0)** |
| `combat_seed.csv` | ComBat / MISPR subset (Atwi and Rajput) for cold-starting the surrogates. | to download |
| `scirep2023_69solvents.csv` | 69 solvents with COSMO-RS ΔmixG (*Sci. Rep.* 2023). | request from the authors |

Minimum columns of `joule2021_table1_solubility.csv`:

```
system_id, solvent, diluent, salt, salt_conc_M, category, donor_number, species, solubility_mM_S, temperature_C, source_note
```

`species` takes the values `S8`, `Li2S8`, `Li2S6`, `Li2S4`, `Li2S2`, `Li2S`. `category` takes `moderately`, `sparingly`, `highly` (solvating), as the review organizes them.

Literature Coulombic-efficiency data, if compiled, go in a separate file with the conditions (S loading, E/S ratio, current density, voltage window, paper) as mandatory columns. They are used only as qualitative rank validation, within a single paper.
