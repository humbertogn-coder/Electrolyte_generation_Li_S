"""Carga y validación de los contratos JSON congelados en `contracts/`."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from pace_s import CONTRACTS_DIR

SCHEMAS = {
    "kmc_interface": CONTRACTS_DIR / "kmc_interface.schema.json",
    "candidate_table": CONTRACTS_DIR / "candidate_table.schema.json",
}


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict[str, Any]:
    path = SCHEMAS[name]
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def validator(name: str) -> Draft202012Validator:
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate(name: str, record: dict[str, Any]) -> None:
    """Lanza `jsonschema.ValidationError` si `record` no cumple el contrato `name`."""
    validator(name).validate(record)


def errors(name: str, record: dict[str, Any]) -> list[str]:
    """Lista legible de violaciones; vacía si el registro es válido."""
    return [
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in validator(name).iter_errors(record)
    ]


def load_and_validate(name: str, path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        record = json.load(fh)
    validate(name, record)
    return record
