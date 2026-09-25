"""M4: oráculos de fidelidad creciente.

tier 1  MLIP-MD (2000-4000 átomos, 2 ns, GPU). Oráculo principal del bucle.
tier 2  DFT sobre finalistas (ωB97X-D o M06-2X / def2-TZVPD / solvatación implícita).
tier 3  AIMD de interfase Li(100)/electrolito (20-40 ps, top 3).

LIMITACIONES (se propagan a `provenance.notes` de cada registro):
- Los MLIP universales NO son reactivos de forma fiable: tier 1 da estructura de
  solvatación y transporte, no mecanismos de descomposición.
- Nernst-Einstein sobreestima la conductividad; usar Green-Kubo o Einstein
  colectivo y reportar el método.
- PBE-D3 sobreestima tasas de descomposición hasta 9 órdenes de magnitud
  (arXiv 2509.14067). Potenciales de reducción y barreras de tier 2 se calculan
  con funcional híbrido, y el nivel de teoría acompaña a cada número.
"""
