"""Tier 0.5: explicit solvation clusters Li+ / solvent / FSI- (PACE-S 2.0).

Small charged and neutral clusters such as [Li(S)]+, [Li(S)2]+, [Li(FSI)] and
[Li(FSI)(S)] are built, optimized at one fixed level of theory, and combined in
balanced reactions (ligand exchange, binding). The resulting energies are
quantum-chemical descriptors of competitive solvation inside a reference
continuum. They are NOT electrolyte properties: they do not give CIP/AGG
populations, desolvation barriers, conductivity or cell lifetime.

Modules
- `species`         species, charges, multiplicity, stoichiometry, cluster protocols
- `reactions`       balanced reactions and reaction energies with consistency checks
- `cluster_builder` initial 3D poses (binding modes, conformers, seeded and reproducible)
- `ensemble`        QC of optimized poses (connectivity, dissociation, coordination)
                    and choice of the minimum
- `runner`          xtb on one pose -> calculation record (resumable)
- `run_exchange`    CLI: exchange and binding energies of a few solvents

Planned (see docs/adr/0001-solvation-al-v2.md): crest_runner, properties.
"""
