"""Minimal xtb (GFN2-xTB) wrapper for tier 0.

Locates the binary (`XTB_BIN` or `xtb` on the PATH), runs it in an isolated
working directory and parses what tier 0 needs: total energy, HOMO/LUMO/gap,
dipole, Gsolv (ALPB), Mulliken charges and, with `--esp`, the surface
electrostatic potential.

Output units: energies in Eh unless the name says otherwise (`*_eV`,
`*_kcal_mol`); dipole in Debye; ESP in kcal/mol.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("pace_s.xtb")

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
    energy_eh: float | None
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
# printed by the ESP routine before it writes the files: max, min, average (Eh)
_RE_ESP_SUMMARY = re.compile(r"maximum/minimum/av ESP value\s*:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def parse_esp_summary(text: str) -> tuple[float, float] | None:
    """(ESPmin, ESPmax) in kcal/mol from the summary line of `xtb --esp`, or None."""
    m = _RE_ESP_SUMMARY.findall(text)
    if not m:
        return None
    vmax, vmin, _ = (float(x) for x in m[-1])
    return vmin * EH_TO_KCAL, vmax * EH_TO_KCAL


def parse_output(text: str, require_normal_termination: bool = True, require_energy: bool = True) -> dict:
    """Extract the fields of a regular `xtb` run. Takes the last occurrence of each."""
    if require_normal_termination and "normal termination of xtb" not in text:
        tail = "\n".join(text.strip().splitlines()[-15:])
        raise XtbError(f"xtb did not terminate normally:\n{tail}")

    def last(rx: re.Pattern, group: int = 1, cast=float):
        m = rx.findall(text)
        if not m:
            return None
        v = m[-1]
        return cast(v[group - 1] if isinstance(v, tuple) else v)

    energy = last(_RE_ENERGY)
    if energy is None and require_energy:
        raise XtbError("no TOTAL ENERGY in the xtb output")
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
    """Run xtb on `xyz` inside `workdir` (created if needed) and return the parsed result."""
    xtb = find_xtb()
    if xtb is None:
        raise XtbError("xtb not found: install it with `conda install -c conda-forge xtb` or set XTB_BIN")
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
        # explicit encoding: on Windows the default is cp1252 and the xtb output
        # contains bytes that codec cannot decode
        proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise XtbError(f"xtb exceeded {timeout_s} s in {workdir}") from e
    # xtb prints "normal termination" to stderr; both streams are parsed together
    text = proc.stdout + "\n" + proc.stderr
    (workdir / "xtb.out").write_text(text, encoding="utf-8")
    # The Windows build of xtb 6.7.1 sometimes crashes inside the ESP routine,
    # i.e. before the final TOTAL ENERGY block and "normal termination". For an
    # --esp run only the ESP matters (the energy comes from the optimization run),
    # so the result is accepted if the ESP can be recovered from xtb_esp.dat or
    # from the summary line the routine prints; otherwise XtbError is raised below.
    fields = parse_output(text, require_normal_termination=not esp, require_energy=not esp)
    if esp and "normal termination of xtb" not in text:
        log.debug("xtb --esp in %s ended without normal termination; recovering what it wrote", workdir)

    res = XtbResult(**fields, workdir=workdir)
    charges_file = workdir / "charges"
    if charges_file.exists():
        res.charges = [float(x) for x in charges_file.read_text().split()]
    opt_xyz = workdir / "xtbopt.xyz"
    res.opt_xyz = opt_xyz if (opt and opt_xyz.exists()) else src
    if esp:
        esp_file = workdir / "xtb_esp.dat"
        if esp_file.exists():
            try:
                data = np.loadtxt(esp_file)
                if data.ndim == 2 and data.shape[1] >= 4 and len(data) > 0:
                    res.esp_min_kcal_mol = float(data[:, 3].min() * EH_TO_KCAL)
                    res.esp_max_kcal_mol = float(data[:, 3].max() * EH_TO_KCAL)
            except ValueError:  # truncated file
                pass
        if res.esp_min_kcal_mol is None:
            summary = parse_esp_summary(text)
            if summary is not None:
                res.esp_min_kcal_mol, res.esp_max_kcal_mol = summary
        if res.esp_min_kcal_mol is None:
            raise XtbError(f"xtb --esp produced no usable ESP in {workdir}")
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
    """Conformer search with CREST; returns the best conformer's xyz (`crest_best.xyz`).

    CREST has no Windows build: on a laptop the RDKit fallback is used
    (`geometry.embed_lowest`). With `quick=True` it costs 1-5 min per small molecule.
    """
    crest = find_crest()
    if crest is None:
        raise XtbError("crest not found (set CREST_BIN or use conformers: rdkit)")
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
    proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout_s)
    (workdir / "crest.out").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    best = workdir / "crest_best.xyz"
    if not best.exists():
        raise XtbError(f"crest did not produce crest_best.xyz in {workdir}")
    return best
