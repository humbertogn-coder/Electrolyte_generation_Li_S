"""PACE-S: pipeline de generación y cribado activo de electrolitos para Li-S / Li-SPAN.

Regla transversal: todo número que salga de este paquete lleva `campaign_id`,
`tier` y `level_of_theory`. Tier 0 solo ordena y filtra; nunca entra al paper.
"""

from pathlib import Path

__version__ = "0.0.1"

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "contracts"
CONFIGS_DIR = REPO_ROOT / "workflows" / "configs"
DATA_DIR = REPO_ROOT / "data"
