# ADR 0001 — PACE-S 2.0: explicit solvation clusters and active learning for Li-SPAN

- Status: accepted (implementation in progress on branch `feature/solvation-al-v2`)
- Date: 2026-11-03
- Baseline: `main` at `dbe06bc`; test report in `docs/baseline/pytest_baseline_dbe06bc.txt`
- Source: "PACE-S 2.0 — Especificación técnica y plan de implementación" (C. Guerrero, 2026-10-31), reviewed 2026-11-01

## Context

PACE-S v1 generates ~60k solvent/diluent candidates by enumerative mutation and
computes Tier-0 descriptors (GFN2-xTB/ALPB(ether)). Its central Li-S descriptor,
the 1:1 solvent–Li2S8 binding energy, failed the solubility gate against Table 1
of Joule 2021 (Spearman −0.40/−0.51 on 14 dilute systems; other proxies dropped
to 0.35–0.59 in leave-one-out; PTB did not change the ranking). In October 2026
the group moved the scientific priority to Li-SPAN, where no solubility table
exists and the only comparable experimental series (PNNL/UCSD, 7 LHCE
formulations) has too few points to train on.

## Decision

1. Evolve PACE-S; do not start a new repository.
2. Add a new campaign (`SPV2-01`, Li-SPAN) in a new configuration file. The
   existing `campaign_LiS.yaml` and `campaign_LiSPAN.yaml` stay unchanged.
3. Add a Tier 0.5 of explicit clusters (Li+, FSI−, solvent S). The first label to
   learn is a quantum-chemical reaction energy, the ligand-exchange energy

       [Li(S)n]+ + FSI−  ->  [Li(FSI)(S)(n−1)] + S

   not cell lifetime, Coulombic efficiency or solubility.
4. Keep `contracts/candidate_table.schema.json` and `kmc_interface.schema.json`
   v1.0.0 byte-identical (pinned by SHA-256 in `tests/test_solvation_contracts.py`).
   New, separately versioned contracts: `cluster_record`, `calculation_record`,
   `selection_record` (all v1.0.0).
5. New identifiers: `PSC-xxxxxx` cluster, `PCC-xxxxxx` calculation,
   `PSL-xxxxxx` selection. `PSM-` (molecule) and `PACES-` (formulation) keep
   their meaning. `PSL-` is not in the source plan; it was added so that
   selection records have their own key.
6. MLIP-MD, AIMD, COSMO-RS and the kMC export remain optional adapters and never
   block the screening workflow. No missing quantity is ever fabricated.

## Additional decisions taken in the review (2026-11-01)

### D1 — Coordination-saturated clusters (denticity bias)

[Li(S)2]+ gives Li+ four donor atoms with DME but two with a monodentate ether
(THP, DBE, EMP) or a diluent (TTE). An exchange energy defined only on n = 2
would mix anion competition with ligand denticity, the same kind of bias
(glyme chelation) that broke the Li2S8 descriptor. Therefore:

- `species.donor_atoms` counts donor atoms (neutral O; amine N not bonded to an
  acyl-like atom; aromatic N without H; thioether S).
- Protocols `li_sat` / `li_fsi_sat` use n = ceil(4 / min(n_donor, 4)) solvent
  molecules (DME 2, THP 4, triglyme 1, TTE 4), so Li+ is offered about four
  donor atoms; the exchange is defined for any n.
- Every cluster record stores `n_donor_atoms_offered`; the coordination actually
  reached is measured after optimization (planned in `ensemble.py`).
- Both the n = 2 exchange (as in the plan) and the saturated exchange are
  computed and compared; models are also evaluated within denticity classes.

### D2 — Exhaustive xTB labels, active learning where it costs

At GFN2-xTB level the cluster set is cheap (order: 500 molecules × 3–4 clusters
× ~8 poses ≈ 10^4 optimizations of 20–60 atoms, tens of CPU-hours). Active
learning cannot save meaningful cost there. The xTB tier is computed
exhaustively for the pilot pool and used for a retrospective benchmark of the
acquisition policies (known ground truth, identical budgets for AL, random and
diversity). Live active learning is reserved for the expensive levels (CREST
ensembles beyond the pilot, DFT).

### D3 — Reuse of existing geometry code

The cluster builder reuses `pace_s.descriptors.geometry` (`coordination_sites`,
`place_ion`, `place_cluster`, `rotation_aligning`), which already places Li+
and a rigid cluster (Li2S8) on donor sites. New work is FSI− sampling:
cis/trans conformers and O-monodentate, O,O-bidentate and N binding modes.

### D4 — CREST only on Grace and only for selected molecules

CREST does not run on Windows. The local builder (multiple poses + xTB) is the
default; CREST/QCG is applied on Grace to ~40 molecules selected by pose-energy
spread or flexibility, after a convergence check on ~10 molecules comparing
both searches.

### D5 — DFT engine is swappable

The DFT step writes an input, runs it, and parses the energy behind one
interface; ORCA is the default (access requested at HPRC, ticket #89871),
Gaussian/PySCF/Psi4 are acceptable alternatives. The program and version are
part of every calculation record (`program.name`, `program.version`). VASP is
not used for isolated charged clusters (periodic charged-cell artifacts, no
range-separated hybrids, no implicit solvent without VASPsol); it remains the
code for interfaces and AIMD.

### D6 — Installation on Windows

The heavy `ml` extra (torch, botorch) is not needed for the MVP. On Anaconda
Prompt use double quotes: `python -m pip install -e ".[chem,dev]"` plus
`scikit-learn` when the surrogate step starts.

### D7 — Sampling convergence, merged runs and CREST as a pose generator (2026-11-04)

The 6-pose demo gave exchange energies that differed by up to 0.9 eV between a
laptop and the cloud (same code and seed, different RDKit builds); with 24 poses
the difference fell to 0.04-0.14 eV, of the same size as the DEE-DME gap. So:

- `convergence.assess` judges every cluster: half_drop_eV (how much the minimum
  fell in the second half of the poses), run_spread_eV (spread of the per-run
  minima when runs from several machines or seeds are merged) and
  sampling_drift_eV = max of both. Converged: drift <= 0.02 eV and, for Li+
  clusters, two accepted poses within 0.05 eV of the minimum. The drift is a
  lower bound of the sampling error, reported with every reaction energy
  (summed over terms with |nu|).
- `merge_runs` merges run directories (refusing mixed levels of theory) and
  exports the best geometry of every non-converged cluster for CREST.
- CREST 3.0.2 (static binary, `scripts/grace/install_crest.sh`) runs on Grace in
  NCI mode with the same GFN2/ALPB(ether) (`workflows/slurm/crest_clusters.sbatch`).
  `crest_import` re-optimizes the lowest CREST conformers with the same xtb
  settings and QC, so CREST is a pose generator and every energy stays at one
  level. A cluster searched by CREST counts as converged; the gain over the
  pose sampling is reported (crest_gain_eV). Cloud check on Li+/FSI-/DME
  (`--quick`, 5.5 min on 2 cores): CREST found the same minimum as the 24 poses
  (gain 0.0003 eV).
- Isolated molecules enter every exchange energy, so they get 60 RDKit
  embeddings and at least 12 optimized conformers (cheap). With the old
  10 embeddings one machine never found the lowest FSI- conformer (0.053 eV).

### D8 — No coordination-saturated clusters for diluents (2026-11-04)

D1 is kept for solvents. For diluents (role `diluent`), [Li(S)4]+ and
[Li(FSI)(S)3] were unphysical in the demo: Li(TTE)4+ ended with no O within
2.6 A (held by F) and TTE detached in many poses; they also took two thirds of
the run time. A diluent is defined by barely solvating Li+, so only exchange_n2
and binding_n1 are computed for it (`run_exchange --saturated auto`; `always`
restores the old behaviour).

## Physical rules enforced in code

- Every reaction checks conservation of elements and total charge on creation
  (`solvation.reactions.Reaction`).
- A reaction energy refuses terms that are missing, failed, or differ in method,
  solvation model, solvation reference, geometry type or energy kind.
- Charge, multiplicity and formula of a cluster are derived from its species.
- A failed calculation is a record with `success=false` and an `error_code`; it
  cannot carry an energy (enforced by the schema).
- dE is an electronic energy difference inside a reference continuum
  ("cluster-in-reference-continuum"); it is not a free energy, a CIP/AGG
  fraction, a desolvation barrier or a lifetime predictor.

## Open questions

- Phosphorus in `allowed_elements` (TEP, PFPN are recommended Li-SPAN seeds).
- Verification of the PNNL/UCSD series against primary sources before it is
  encoded as an external holdout.
- DFT benchmark level: wB97X-D3/def2-TZVPD with def2/J, RIJCOSX and CPCM
  (epsilon 4.33, refrac 1.3526) runs on Grace with ORCA 6.1.1 (access granted
  2026-11-04, HPRC ticket #89871; `workflows/slurm/orca_check.sbatch`,
  `docs/baseline/orca_check_grace_20261104.txt`). Basis-set dependence and BSSE
  are still to be checked.
- Implicit solvation of the bare Li+ ion differs strongly between continuum
  models (ALPB vs CPCM); binding energies Li+ + S -> [Li(S)]+ inherit that error,
  exchange energies do not contain bare Li+ and should be more transferable
  between levels. To be tested in the xTB/DFT comparison.

## Consequences

The Tier-0 code, its columns and the negative Li2S8 result remain as they are.
New functionality lives in `pace_s.solvation`, `pace_s.surrogate`,
`pace_s.acquire` and new contracts, each with tests that run without xtb,
CREST or ORCA (markers `slow`, `external`, `hpc` separate the heavy ones).
