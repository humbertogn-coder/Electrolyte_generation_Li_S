"""M4: oracles of increasing fidelity.

tier 1  MLIP-MD (2000-4000 atoms, 2 ns, GPU). Main oracle of the loop.
tier 2  DFT on finalists (wB97X-D or M06-2X / def2-TZVPD / implicit solvation).
tier 3  AIMD of the Li(100)/electrolyte interface (20-40 ps, top 3).

LIMITATIONS (propagated to `provenance.notes` of every record):
- Universal MLIPs are NOT reliably reactive: tier 1 gives solvation structure
  and transport, not decomposition mechanisms.
- Nernst-Einstein overestimates conductivity; use Green-Kubo or the collective
  Einstein relation and report the method.
- PBE-D3 overestimates decomposition rates by up to nine orders of magnitude
  (arXiv 2509.14067). Tier-2 reduction potentials and barriers are computed
  with a hybrid functional, and the level of theory accompanies every number.
"""
