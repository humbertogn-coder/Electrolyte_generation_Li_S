"""M2: tier 0, cheap descriptors (< 5 CPU-min per molecule).

Wrappers around xtb (GFN2-xTB or g-xTB), CREST, morfeus and RDKit. Produces
the `tier0.*` columns of the `candidate_table` contract.

RULE: tier 0 is used only to rank and filter. No value from this layer goes
into the paper. Every reported number comes from tier 1 or higher.

Central descriptor: `li2s8_binding_eV` (solvent-Li2S8 interaction energy),
calibrated against Table 1 of Joule 2021 at the H3 gate.
"""
