"""Wrapper mínimo de xtb (GFN2-xTB) para tier 0.

Localiza el binario (`XTB_BIN` o `xtb` en el PATH), lo ejecuta en un directorio
de trabajo aislado y parsea lo que tier 0 necesita: energía total, HOMO/LUMO/gap,
dipolo, Gsolv (ALPB), cargas de Mulliken y, con `--esp`, el potencial
electrostático en la superficie.

Unidades de salida: energías en Eh salvo que el nombre diga lo contrario
(`*_eV`, `*_kcal_mol`); dipolo en Debye; ESP en kcal/mol.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EH_TO_EV = 27.211386245988
EH_TO_KCAL = 627.5094740631


class XtbError(RuntimeError):
    pass


def find_xtb() -> str | None:
    return os.environ.get("XTB_BIN") or shutil.which("xtb")


def find_crest() -> str | None:
    return os.environ.get("CREST_BIN") or shutil.which("crest")


@dataclass
class XtbResult:
    energy_eh: float
    homo_eV: float | None = None
    lumo_eV: float | None = None
    gap_eV: float | None = None
    dipole_D: float | None = None
    gsolv_eh: float | None = None
    charges: list[float] = field(default_factory=list)
    opt_xyz: Path | None = None
    esp_min_kcal_mol: float | None = None
    esp_max_kcal_mol: float | None = None
    workdir: Path | None = None


_RE_ENERGY = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
_RE_GAP = re.compile(r"HOMO-LUMO GAP\s+(-?\d+\.\d+)\s+eV")
_RE_HOMO = re.compile(r"(-?\d+\.\d+)\s+\(HOMO\)")
_RE_LUMO = re.compile(r"(-?\d+\.\d+)\s+\(LUMO\)")
_RE_GSOLV = re.compile(r"-> Gsolv\s+(-?\d+\.\d+)\s+Eh")
_RE_DIPOLE = re.compile(r"molecular dipole:.*?full:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", re.S)


def parse_output(text: str) -> dict:
    """Extrae los campos de un `xtb` normal. Toma la última ocurrencia de cada uno."""
    if "normal termination of xtb" not in text:
        tail = "\n".join(text.strip().splitlines()[-15:])
        raise XtbError(f"xtb no terminó normalmente:\n{tail}")

    def last(rx: re.Pattern, group: int = 1, cast=float):
        m = rx.findall(text)
        if not m:
            return None
        v = m[-1]
        return cast(v[group - 1] if isinstance(v, tuple) else v)

    energy = last(_RE_ENERGY)
    if energy is None:
        raise XtbError("sin TOTAL ENERGY en la salida de xtb")
    return {
        "energy_eh": energy,
        "gap_eV": last(_RE_GAP),
        "homo_eV": last(_RE_HOMO),
        "lumo_eV": last(_RE_LUMO),
        "gsolv_eh": last(_RE_GSOLV),
        "dipole_D": last(_RE_DIPOLE, group=4),
    }


def run_xtb(
    xyz: str | Path,
    workdir: str | Path,
    *,
    charge: int = 0,
    uhf: int = 0,
    opt: bool = True,
    gfn: int = 2,
    alpb: str | None = "ether",
    esp: bool = False,
    threads: int | None = None,
    timeout_s: int = 900,
    extra: list[str] | None = None,
) -> XtbResult:
    """Ejecuta xtb sobre `xyz` dentro de `workdir` (se crea) y devuelve el resultado parseado."""
    xtb = find_xtb()
    if xtb is None:
        raise XtbError("no se encontró xtb: instala `conda install -c conda-forge xtb` o define XTB_BIN")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    xyz = Path(xyz).resolve()
    src = workdir / "input.xyz"
    if src.resolve() != xyz:
        shutil.copy(xyz, src)

    cmd = [xtb, "input.xyz", "--gfn", str(gfn), "--chrg", str(charge)]
    if uhf:
        cmd += ["--uhf", str(uhf)]
    if opt:
        cmd += ["--opt"]
    if alpb:
        cmd += ["--alpb", alpb]
    if esp:
        cmd += ["--esp"]
    if extra:
        cmd += list(extra)

    env = dict(os.environ)
    n = threads or int(env.get("OMP_NUM_THREADS", "1"))
    env.update({"OMP_NUM_THREADS": str(n), "MKL_NUM_THREADS": str(n), "OMP_STACKSIZE": env.get("OMP_STACKSIZE", "1G")})
    try:
        proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise XtbError(f"xtb excedió {timeout_s} s en {workdir}") from e
    # xtb escribe "normal termination" en stderr; se parsea todo junto
    text = proc.stdout + "\n" + proc.stderr
    (workdir / "xtb.out").write_text(text, encoding="utf-8")
    fields = parse_output(text)

    res = XtbResult(**fields, workdir=workdir)
    charges_file = workdir / "charges"
    if charges_file.exists():
        res.charges = [float(x) for x in charges_file.read_text().split()]
    opt_xyz = workdir / "xtbopt.xyz"
    res.opt_xyz = opt_xyz if (opt and opt_xyz.exists()) else src
    if esp:
        esp_file = workdir / "xtb_esp.dat"
        if esp_file.exists():
            data = np.loadtxt(esp_file)
            if data.ndim == 2 and data.shape[1] >= 4:
                res.esp_min_kcal_mol = float(data[:, 3].min() * EH_TO_KCAL)
                res.esp_max_kcal_mol = float(data[:, 3].max() * EH_TO_KCAL)
    return res


def run_crest(
    xyz: str | Path,
    workdir: str | Path,
    *,
    charge: int = 0,
    gfn: int = 2,
    alpb: str | None = "ether",
    quick: bool = True,
    threads: int | None = None,
    timeout_s: int = 3600,
) -> Path:
    """Búsqueda de confórmeros con CREST; devuelve el xyz del mejor confórmero (`crest_best.xyz`).

    CREST no tiene build para Windows: en la laptop se usa el fallback de RDKit
    (`geometry.embed_lowest`). Con `quick=True` cuesta 1-5 min por molécula pequeña.
    """
    crest = find_crest()
    if crest is None:
        raise XtbError("no se encontró crest (define CREST_BIN o usa conformers: rdkit)")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / "input.xyz"
    shutil.copy(xyz, src)
    cmd = [crest, "input.xyz", f"--gfn{gfn}", "--chrg", str(charge), "--noreftopo"]
    if alpb:
        cmd += ["--alpb", alpb]
    if quick:
        cmd += ["--quick"]
    env = dict(os.environ)
    n = threads or int(env.get("OMP_NUM_THREADS", "1"))
    cmd += ["-T", str(n)]
    env.update({"OMP_NUM_THREADS": str(n), "OMP_STACKSIZE": env.get("OMP_STACKSIZE", "1G")})
    proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout_s)
    (workdir / "crest.out").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    best = workdir / "crest_best.xyz"
    if not best.exists():
        raise XtbError(f"crest no produjo crest_best.xyz en {workdir}")
    return best
