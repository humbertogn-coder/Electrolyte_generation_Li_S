"""Provenance helpers: stable hashes of inputs and files.

`stable_hash` gives the same SHA-256 for the same content regardless of key
order or float formatting noise below 1e-10, so an `input_hash` identifies a
calculation's inputs (geometry, method, charge, parameters) and lets a runner
skip work that already succeeded.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _normalize(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError("NaN/inf cannot be hashed as an input")
        return round(obj, 10) + 0.0  # +0.0 turns -0.0 into 0.0
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj


def stable_hash(obj: Any) -> str:
    """SHA-256 (hex) of a JSON-serializable object, independent of dict key order."""
    text = json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
