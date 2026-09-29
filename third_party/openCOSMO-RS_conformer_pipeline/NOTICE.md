# openCOSMO-RS conformer pipeline (vendored)

Source: https://github.com/TUHH-TVT/openCOSMO-RS_conformer_pipeline, commit `14b3c41`
(master, 2026-09), LGPL-3.0 (see `LICENSE`). Authors: Thomas Gerlach, Simon Müller
et al. (TU Hamburg, Institute of Thermal Separation Processes).

Why vendored: the upstream file is imported as a script (no package), and the
commit above does not parse (three `try:` blocks without `except`, left by the
"fix-executable-finder" merge). Keeping a working copy here makes the Grace runs
reproducible.

Local changes to `ConformerGenerator.py` (all cosmetic, no change in the protocol):

- `find_balloon_executable`, `ORCA.__init__` (orca and otool_xtb lookup): removed
  the dangling `try:` statements; `shutil.which` is called directly.

`cpcm_radii.inp`: upstream file plus lithium, `3  2.13` Å. Upstream has no radius
for Li. 2.13 Å is 1.17 x the Bondi van der Waals radius (1.82 Å), the rule COSMO
uses for elements without an optimized radius; it is consistent with the S (2.16),
Cl (2.05) and F (1.72) values already in the file.

Protocol implemented by the pipeline (openCOSMO-RS 24a, Gerlach et al., Fluid Phase
Equilib. 2024, doi 10.1016/j.fluid.2024.114250):

1. RDKit ETKDG conformers (fallback balloon), RMS 1.0 Å pruning.
2. gas phase: BP86/def2-TZVP(-f) optimization of all conformers, keep the lowest,
   BP86/def2-TZVP optimization + def2-TZVPD single point.
3. CPCM branch: XTB2/ALPB(water) optimization through ORCA (`otool_xtb`), 6 kcal/mol
   window, up to 3 conformers; BP86/def2-TZVP(-f) CPCM optimization, keep 1;
   BP86/def2-TZVP CPCM optimization + def2-TZVPD CPCM single point with the radii
   above and `cut_area 0.01 Å²`.
4. `<name>/COSMO_TZVPD/<name>_c000.orcacosmo` = energy, dipole, geometry, COSMO
   surface (and the ORCA 6.1 `cpcm_corr` conductor correction), read by
   `opencosmorspy`.

Requirements on the machine that runs it: ORCA >= 6.1 on `PATH`, `otool_xtb` on
`PATH` (an xtb binary renamed or symlinked), RDKit. See
`workflows/slurm/cosmors_orca_serial.sbatch`.
