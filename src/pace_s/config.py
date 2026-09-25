"""Configuración de campaña.

Los pesos de los objetivos y las restricciones del espacio de diseño viven en
`workflows/configs/campaign_*.yaml`. Cambiar de Li-S a Li-SPAN es cambiar de
archivo, no de código. Un YAML puede declarar `inherit: <otro.yaml>`; las claves
del hijo se fusionan sobre las del padre (fusión profunda de diccionarios).
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
        raise ValueError(f"Configuración incompleta, faltan: {missing}")
    if cfg["system"] not in ("Li-S", "Li-SPAN"):
        raise ValueError(f"system debe ser Li-S o Li-SPAN, no {cfg['system']!r}")
    objs = set(cfg["objectives"])
    if objs != set(OBJECTIVE_KEYS):
        raise ValueError(f"Los objetivos deben ser exactamente {OBJECTIVE_KEYS}; hay {sorted(objs)}")


def objective_weights(cfg: dict[str, Any]) -> dict[str, float]:
    """Pesos de los objetivos de Pareto (O5 es restricción y no tiene peso)."""
    return {
        k: float(v["weight"])
        for k, v in cfg["objectives"].items()
        if v.get("role") != "constraint"
    }


def dominant_objective(cfg: dict[str, Any]) -> str:
    weights = objective_weights(cfg)
    return max(weights, key=weights.get)
