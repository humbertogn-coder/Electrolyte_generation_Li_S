"""Balanced reactions between solvation clusters and their reaction energies.

Two families are defined:

    binding:   Li+ + n S                ->  [Li(S)n]+
    exchange:  [Li(S)n]+ + FSI-          ->  [Li(FSI)(S)(n-1)] + S

For n = 2 the exchange is the reaction of the plan,

    dE_ex = E([Li(FSI)(S)]) + E(S) - E([Li(S)2]+) - E(FSI-).

Every reaction checks conservation of elements and of total charge when it is
created. `reaction_energy` refuses to combine energies that are missing,
failed, or computed with different methods, environments, geometry types or
energy kinds: a missing value is an error, never a zero.

dE here is an electronic energy difference inside one model and one reference
continuum. A negative dE_ex favours the contact-ion side within that model; it
is not a free energy, a CIP fraction or a population.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from pace_s.descriptors.xtb import EH_TO_EV, EH_TO_KCAL
from pace_s.solvation.species import Composition, Species, fsi_anion, li_cation

# fields that must be identical for every term of one reaction energy
CONSISTENCY_KEYS = ("method", "solvation_model", "solvation_reference", "geometry_type", "energy_kind")


class ReactionBalanceError(ValueError):
    """Elements or charge are not conserved."""


class MissingEnergyError(KeyError):
    """A term of the reaction has no energy record, or its calculation failed."""


class InconsistentEnergiesError(ValueError):
    """Terms were computed with different methods, environments or energy kinds."""


Term = tuple[Composition, int]


@dataclass(frozen=True)
class Reaction:
    name: str
    reactants: tuple[Term, ...]
    products: tuple[Term, ...]

    def __post_init__(self) -> None:
        for _, nu in self.reactants + self.products:
            if not isinstance(nu, int) or nu < 1:
                raise ReactionBalanceError(f"{self.name}: stoichiometric coefficients must be positive integers")
        problems = self.imbalance()
        if problems:
            raise ReactionBalanceError(f"{self.name}: not balanced: {problems}")

    @staticmethod
    def _side(terms: tuple[Term, ...]) -> tuple[Counter, int]:
        elements: Counter = Counter()
        charge = 0
        for comp, nu in terms:
            for el, k in comp.formula.items():
                elements[el] += k * nu
            charge += comp.net_charge * nu
        return elements, charge

    def imbalance(self) -> dict[str, Any]:
        """{} if balanced; otherwise the element and charge differences (products - reactants)."""
        er, qr = self._side(self.reactants)
        ep, qp = self._side(self.products)
        diff = {el: ep.get(el, 0) - er.get(el, 0) for el in set(er) | set(ep)}
        out: dict[str, Any] = {el: d for el, d in sorted(diff.items()) if d}
        if qp != qr:
            out["charge"] = qp - qr
        return out

    def terms(self) -> list[tuple[Composition, int]]:
        """Signed terms: products positive, reactants negative."""
        return [(c, nu) for c, nu in self.products] + [(c, -nu) for c, nu in self.reactants]

    def labels(self) -> list[str]:
        return sorted({c.label for c, _ in self.reactants + self.products})

    def __str__(self) -> str:
        def side(ts):
            return " + ".join((f"{nu} " if nu > 1 else "") + c.label for c, nu in ts)
        return f"{side(self.reactants)} -> {side(self.products)}"


def _isolated(sp: Species) -> Composition:
    return Composition(((sp, 1),))


def binding_reaction(solvent: Species, n: int = 1) -> Reaction:
    """Li+ + n S -> [Li(S)n]+"""
    li = li_cation()
    cluster = Composition(((li, 1), (solvent, n)))
    return Reaction(
        name=f"binding_{solvent.name}_n{n}",
        reactants=((_isolated(li), 1), (_isolated(solvent), n)),
        products=((cluster, 1),),
    )


def exchange_reaction(solvent: Species, n_ligands: int = 2) -> Reaction:
    """[Li(S)n]+ + FSI- -> [Li(FSI)(S)(n-1)] + S   (n >= 1; n = 2 is the plan's reaction)."""
    if n_ligands < 1:
        raise ValueError("n_ligands must be >= 1")
    li, fsi = li_cation(), fsi_anion()
    left = Composition(((li, 1), (solvent, n_ligands)))
    right_parts: list[tuple[Species, int]] = [(li, 1), (fsi, 1)]
    if n_ligands > 1:
        right_parts.append((solvent, n_ligands - 1))
    right = Composition(tuple(right_parts))
    return Reaction(
        name=f"exchange_{solvent.name}_n{n_ligands}",
        reactants=((left, 1), (_isolated(fsi), 1)),
        products=((right, 1), (_isolated(solvent), 1)),
    )


def reaction_energy(reaction: Reaction, energies: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Reaction energy from energy records keyed by composition label.

    Each record needs at least: success (true), energy_eh (number) and the
    fields in CONSISTENCY_KEYS, which must be identical across all terms. The
    `calculation_id`s used are returned for provenance.
    """
    used: dict[str, Mapping[str, Any]] = {}
    for comp, _ in reaction.terms():
        rec = energies.get(comp.label)
        if rec is None:
            raise MissingEnergyError(f"{reaction.name}: no energy record for {comp.label}")
        if rec.get("success") is not True or not isinstance(rec.get("energy_eh"), (int, float)):
            raise MissingEnergyError(
                f"{reaction.name}: calculation for {comp.label} is not a successful energy "
                f"(success={rec.get('success')!r}, error_code={rec.get('error_code')!r})"
            )
        charge, mult = rec.get("charge"), rec.get("multiplicity")
        if charge is not None and charge != comp.net_charge:
            raise InconsistentEnergiesError(f"{comp.label}: record charge {charge} != composition charge {comp.net_charge}")
        if mult is not None and mult != comp.multiplicity:
            raise InconsistentEnergiesError(f"{comp.label}: record multiplicity {mult} != {comp.multiplicity}")
        used[comp.label] = rec

    first = next(iter(used.values()))
    for key in CONSISTENCY_KEYS:
        values = {lbl: r.get(key) for lbl, r in used.items()}
        if len(set(values.values())) > 1:
            raise InconsistentEnergiesError(f"{reaction.name}: '{key}' differs between terms: {values}")

    de_eh = sum(nu * float(used[comp.label]["energy_eh"]) for comp, nu in reaction.terms())
    return {
        "reaction": reaction.name,
        "equation": str(reaction),
        "delta_e_eh": de_eh,
        "delta_e_eV": de_eh * EH_TO_EV,
        "delta_e_kcal_mol": de_eh * EH_TO_KCAL,
        **{key: first.get(key) for key in CONSISTENCY_KEYS},
        "calculation_ids": sorted(str(r.get("calculation_id")) for r in used.values() if r.get("calculation_id")),
    }
