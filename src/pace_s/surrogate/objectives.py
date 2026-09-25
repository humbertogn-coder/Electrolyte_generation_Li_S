"""Transformación de propiedades crudas a objetivos O1-O5 en dirección 'mayor es mejor'.

Los objetivos son Pareto, no se suman. Los pesos del YAML solo entran en la
ponderación de referencia de qNEHVI y en el ordenamiento final.
"""

from __future__ import annotations


def to_objectives(record: dict, cfg: dict) -> dict[str, float]:
    raise NotImplementedError("Se implementa en semana 4 con los primeros datos de tier 1.")
