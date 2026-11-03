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
- DFT benchmark level (wB97X-D3/def2-TZVPD is the starting point; check
  syntax/availability in ORCA 6.1.1 and basis-set dependence, BSSE).

## Consequences

The Tier-0 code, its columns and the negative Li2S8 result remain as they are.
New functionality lives in `pace_s.solvation`, `pace_s.surrogate`,
`pace_s.acquire` and new contracts, each with tests that run without xtb,
CREST or ORCA (markers `slow`, `external`, `hpc` separate the heavy ones).
