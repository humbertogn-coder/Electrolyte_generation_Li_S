"""M3 (b): multi-objective acquisition and active-learning loop management.

Acquisition: qNEHVI (BoTorch). A deliberate difference from the single-objective
Expected Improvement of Nat. Commun. 2025; it avoids arbitrary weights.
Batches of 60 formulations, 3 iterations. Iteration 1 seeded by diversity and
by ComBat. The comparison against random selection is mandatory (H4).
"""
