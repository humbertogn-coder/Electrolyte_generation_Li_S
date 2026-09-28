# PACE-S

**P**ipeline for **A**ctive generation and s**C**reening of **E**lectrolytes for lithium-metal batteries with **S**ulfur chemistry (Li-S, with transfer to Li-SPAN).

Balbuena Group, Texas A&M University. Independent of the 3D kinetic Monte Carlo morphology model (`kmc3d`): PACE-S may depend on `kmc3d`; `kmc3d` never depends on PACE-S.

Positioning in one sentence: the map of known Li-S solvents exists; what is missing is the machine that proposes unknown ones and carries them all the way to morphology.

## What it does

```
M0  Seeds, design space and calibration data
M1  Fragment-based generator and mutations     -->  20k-60k SMILES
M2  Tier 0: filters + cheap descriptors        -->  ~3k survivors
M3  Surrogate models + qNEHVI acquisition      <--+
M4  Tier 1: MLIP-MD (main oracle)              ---+  active-learning loop
    Tier 2: DFT (finalists)
    Tier 3: AIMD (top 3, mechanism)
M5  SHAP + parameter export                    -->  3D kMC morphology model
```

## Layout

```
contracts/          frozen JSON schemas (week 0) and examples
src/pace_s/
  generate/         seeds, mutation operators, hard and synthesizability filters
  descriptors/      tier 0: xtb, CREST, morfeus, RDKit
  oracles/          tier 1 MLIP-MD, tier 2 DFT, tier 3 AIMD
  surrogate/        surrogate models with uncertainty
  acquire/          qNEHVI (BoTorch), loop management
  analysis/         SHAP, Pareto, figures
  export/           own format + kmc3d adapter
workflows/
  slurm/            job-array (CPU) and GPU templates for Grace
  configs/          one YAML per campaign; Li-S vs Li-SPAN weights live here
data/
  manifests/        per-campaign indices and checksums (NO trajectories)
  reference/        Joule 2021 Table 1, ComBat, reference molecules
tests/
notebooks/
```

## Installation

```bash
conda env create -f environment.yml          # laptop: generation, surrogates, AL, tests
conda activate pace-s
pip install -e .
pytest
```

To run tier 0 on a laptop (small tests) add xtb to the main environment; it does have a Windows build:

```bash
conda install -c conda-forge xtb
python -m pace_s.descriptors.run_tier0 --config workflows/configs/campaign_LiS.yaml \
    --smiles COCCOC C1COCO1 --workdir data/campaigns/test/tier0 --out data/campaigns/test/tier0_test.csv
```

The full tier-0 campaign runs as a SLURM job array on Grace and is merged back into the candidate table afterwards:

```bash
python -m pace_s.generate.run --config workflows/configs/campaign_LiS.yaml --out data/campaigns/AL-01/candidates_stage1.csv
sbatch --array=0-1199%200 workflows/slurm/tier0_xtb_array.sbatch data/campaigns/AL-01/candidates_stage1.csv
python -m pace_s.descriptors.merge_tier0 --candidates data/campaigns/AL-01/candidates_stage1.csv \
    --blocks data/campaigns/AL-01/tier0 --out data/campaigns/AL-01/candidates_tier0.csv
# candidates_tier0.missing.csv lists what still needs tier 0; feed it back to the job array
```

Three environments, because not everything has a Windows build or resolves together:

| File | Where | Purpose |
|---|---|---|
| `environment.yml` | laptop and Grace | generation, filters, surrogates, AL, analysis, tests |
| `environment-tier0.yml` | Grace (Linux) | xtb, CREST, morfeus: tier-0 descriptors |
| `environment-mlip.yml` | Grace (GPU) | MACE / fairchem (UMA), LAMMPS with plugin: tier 1 and the H2 benchmark |

On Grace, LAMMPS with the MACE or DeePMD plugin is compiled separately; ask for the route the group already solved before compiling from scratch.

## Working rules

1. **Versioned manifests.** One CSV per campaign in `data/manifests/` with id, SMILES, config hash, Grace path and checksum. Git keeps the index, Grace keeps the bytes.
2. **`campaign_id` on every output.** Without it, three loop iterations are unreconstructible.
3. **Provenance on every number.** Tier and level of theory, always. No tier-0 value goes into the paper.
4. **Tests from the first commit.** Valid SMILES, JSON that validates against its schema, reference molecules with expected tier-0 values.
5. **Limitations written inside the code.** Nernst-Einstein overestimates conductivity; MLIPs are not reactive; PBE-D3 overestimates decomposition rates by up to nine orders of magnitude (reduction potentials are computed with a hybrid functional).

## Frozen contracts

- `contracts/kmc_interface.schema.json`: what PACE-S hands to the 3D kMC. The three `kinetics` fields are the kMC's mandatory input.
- `contracts/candidate_table.schema.json`: one row of the master candidate table.

Changing a contract requires bumping `schema_version` and updating the adapter `src/pace_s/export/kmc3d_adapter.py`, which is pinned to a specific `kmc3d` version.

## Switching systems

Li-S and Li-SPAN share the code. They differ in `workflows/configs/campaign_LiS.yaml` and `campaign_LiSPAN.yaml`: objective weights (O2, polysulfide solubility, dominates in Li-S; O3, reductive passivation, dominates in Li-SPAN).

## Related documents

Proposal v2 (`propuesta_pipeline_electrolitos_LiS.md`) and literature review v2 (`revision_literatura_generacion_electrolitos.md`), in the project's parent folder.

## License

To be decided before the repository goes public.
