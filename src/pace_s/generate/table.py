"""Tabla maestra de candidatos: registros anidados (contrato `candidate_table`)
y su forma aplanada en columnas con punto (`generation.stage`, `filters.sascore`).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from pace_s.contracts import errors

SCHEMA_VERSION = "1.0.0"


def candidate_id(n: int) -> str:
    return f"PSM-{n:06d}"


def flatten(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in record.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return isinstance(v, float) and math.isnan(v)
    except TypeError:
        return False


def unflatten(row: dict[str, Any]) -> dict[str, Any]:
    """Inversa de `flatten`. Las columnas vacías (None/NaN) se omiten, salvo las
    que el contrato declara nullable (`generation.parent_id`, `generation.operator`,
    `filters.hard_fail_reason`)."""
    nullable = {"generation.parent_id", "generation.operator", "filters.hard_fail_reason"}
    out: dict[str, Any] = {}
    for key, v in row.items():
        if _is_missing(v):
            if key not in nullable:
                continue
            v = None
        parts = key.split(".")
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = _py(v)
    return out


def _py(v: Any) -> Any:
    """Convierte escalares de numpy/pandas a tipos nativos para jsonschema."""
    if hasattr(v, "item"):
        v = v.item()
    return v


def validate_frame(df: pd.DataFrame, max_errors: int = 20) -> list[str]:
    """Valida cada fila contra el contrato. Devuelve mensajes (vacío si todo es válido)."""
    msgs: list[str] = []
    for i, row in enumerate(df.to_dict(orient="records")):
        errs = errors("candidate_table", unflatten(row))
        for e in errs:
            msgs.append(f"row {i} ({row.get('candidate_id')}): {e}")
            if len(msgs) >= max_errors:
                return msgs
    return msgs
