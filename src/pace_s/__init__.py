"""PACE-S: pipeline for active generation and screening of Li-S / Li-SPAN electrolytes.

Cross-cutting rule: every number leaving this package carries `campaign_id`,
`tier` and `level_of_theory`. Tier 0 only ranks and filters; it never enters the paper.
"""

from pathlib import Path

__version__ = "0.0.1"

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "contracts"
CONFIGS_DIR = REPO_ROOT / "workflows" / "configs"
DATA_DIR = REPO_ROOT / "data"
