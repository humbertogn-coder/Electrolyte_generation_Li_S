"""Master candidate table: nested records (`candidate_table` contract) and
their flattened form with dot-separated columns (`generation.stage`, `filters.sascore`).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from pace_s.contracts import errors

SCHEMA_VERSION = "1.0.0"

# Columns that must stay strings when reading CSV: pandas would otherwise turn
# a commit hash like "0000000" or "1234567" into an integer.
STRING_COLUMNS = [
    "schema_version", "candidate_id", "smiles", "canonical_smiles", "inchikey", "role", "status",
    "campaign_id", "commit", "generation.parent_id", "generation.operator", "filters.hard_fail_reason",
    "tier0.method", "tier0.error", "tier1.electrolyte_id", "tier1.mlip", "tier2.level_of_theory",
]


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a candidate table (CSV or parquet) with the string columns typed correctly."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={c: str for c in STRING_COLUMNS})


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
    """Inverse of `flatten`. Empty columns (None/NaN) are dropped, except those
    the contract declares nullable (`generation.parent_id`, `generation.operator`,
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
    """Convert numpy/pandas scalars to native types for jsonschema."""
    if hasattr(v, "item"):
        v = v.item()
    return v


def validate_frame(df: pd.DataFrame, max_errors: int = 20) -> list[str]:
    """Validate every row against the contract. Returns messages (empty if all valid)."""
    msgs: list[str] = []
    for i, row in enumerate(df.to_dict(orient="records")):
        errs = errors("candidate_table", unflatten(row))
        for e in errs:
            msgs.append(f"row {i} ({row.get('candidate_id')}): {e}")
            if len(msgs) >= max_errors:
                return msgs
    return msgs
