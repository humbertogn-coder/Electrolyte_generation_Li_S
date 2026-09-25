"""M2: tier 0, descriptores baratos (< 5 min CPU por molécula).

Wrappers sobre xtb (GFN2-xTB o g-xTB), CREST, morfeus y RDKit. Produce las
columnas `tier0.*` del contrato `candidate_table`.

REGLA: tier 0 solo se usa para ordenar y filtrar. Ningún valor de esta capa
entra al paper. Todo número reportado viene de tier 1 o superior.

Descriptor central: `li2s8_binding_eV` (energía de interacción solvente-Li2S8),
calibrado contra la Tabla 1 de Joule 2021 en la compuerta H3.
"""
