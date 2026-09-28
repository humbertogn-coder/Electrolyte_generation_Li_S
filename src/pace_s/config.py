"""Campaign configuration.

Objective weights and design-space constraints live in
`workflows/configs/campaign_*.yaml`. Switching from Li-S to Li-SPAN means
switching files, not code. A YAML may declare `inherit: <other.yaml>`; the
child's keys are merged over the parent's (deep dictionary merge).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from pace_s import CONFIGS_DIR

OBJECTIVE_KEYS = (
    "O1_weak_solvation",
    "O2_polysulfide_retention",
    "O3_reductive_passivation",
    "O4_transport",
    "O5_synthesizability",
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_campaign(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIGS_DIR / path
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_campaign(path.parent / parent)
        cfg = _deep_merge(base, cfg)
    _check(cfg)
    return cfg


def _check(cfg: dict[str, Any]) -> None:
    missing = [k for k in ("campaign_id", "system", "objectives", "design_space") if k not in cfg]
    if missing:
        raise ValueError(f"Incomplete configuration, missing: {missing}")
    if cfg["system"] not in ("Li-S", "Li-SPAN"):
        raise ValueError(f"system must be Li-S or Li-SPAN, not {cfg['system']!r}")
    objs = set(cfg["objectives"])
    if objs != set(OBJECTIVE_KEYS):
        raise ValueError(f"Objectives must be exactly {OBJECTIVE_KEYS}; got {sorted(objs)}")


def objective_weights(cfg: dict[str, Any]) -> dict[str, float]:
    """Weights of the Pareto objectives (O5 is a constraint and has no weight)."""
    return {
        k: float(v["weight"])
        for k, v in cfg["objectives"].items()
        if v.get("role") != "constraint"
    }


def dominant_objective(cfg: dict[str, Any]) -> str:
    weights = objective_weights(cfg)
    return max(weights, key=weights.get)
