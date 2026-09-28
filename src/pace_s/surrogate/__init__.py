"""M3 (a): surrogate models with uncertainty.

Representation: tier-0 descriptors + reduced Morgan fingerprints + formulation
variables. Models: gradient-boosting ensemble with quantile uncertainty, plus a
Gaussian process for the acquisition. Chemprop as reference. Cold start seeded
with ComBat.

Every prediction is reported with `distance_to_seeds` (1 - maximum Tanimoto
similarity to the seeds): the generator extrapolates, and we say by how much.
"""
