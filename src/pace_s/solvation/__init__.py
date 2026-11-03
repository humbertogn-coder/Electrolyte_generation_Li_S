"""Tier 0.5: explicit solvation clusters Li+ / solvent / FSI- (PACE-S 2.0).

Small charged and neutral clusters such as [Li(S)]+, [Li(S)2]+, [Li(FSI)] and
[Li(FSI)(S)] are built, optimized at one fixed level of theory, and combined in
balanced reactions (ligand exchange, binding). The resulting energies are
quantum-chemical descriptors of competitive solvation inside a reference
continuum. They are NOT electrolyte properties: they do not give CIP/AGG
populations, desolvation barriers, conductivity or cell lifetime.

Modules
- `species`   species, charges, multiplicity, stoichiometry, cluster protocols
- `reactions` balanced reactions and reaction energies with consistency checks

Planned (see docs/adr/0001-solvation-al-v2.md): cluster_builder, crest_runner,
ensemble, properties, runner.
"""
