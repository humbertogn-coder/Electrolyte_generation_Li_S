"""M1: design-space generation.

Stage 1: deterministic enumerative operators on the seeds (close, auditable
variations). Stage 2: open mutation with SAFE / REINVENT4 on the best of
stage 1. Both go through `filters.hard_filter` and the synthesizability filter
(SAScore < 4.5, RAScore > 0.7).
"""
