"""Tier 0: cheap per-molecule descriptors with GFN2-xTB (M2).

    # a few molecules, for testing
    python -m pace_s.descriptors.run_tier0 --config workflows/configs/campaign_LiS.yaml \
        --smiles COCCOC C1COCO1 --workdir data/campaigns/test/tier0 --out tier0_test.csv

    # one block of the candidate table (this is how the SLURM job array calls it)
    python -m pace_s.descriptors.run_tier0 --config ... --candidates candidates_stage1.csv \
        --start 0 --n 50 --workdir $SCRATCH/pace-s/AL-01/tier0 --out tier0_000000.csv

Per molecule (target < 5 CPU-min):
  1. RDKit 2D descriptors (mass, TPSA, logP, fluorination, C/O) and 3D volume;
  2. conformer: RDKit ETKDG + MMFF (default) or CREST if `tiers.tier0.conformers: CREST`;
  3. xtb --opt --alpb <solvent>: energy, HOMO/LUMO/gap, dipole, Gsolv, charges;
  4. xtb --esp on the optimized geometry: ESPmin / ESPmax;
  5. Li+ binding energy: Li+ placed on each O/N/S site (up to 4, by charge),
     optimized, and the minimum of E(complex) - E(mol) - E(Li+) is taken;
  6. Li2S8 affinity: same, anchoring one Li of the reference Li2S8 cluster
     (`data/reference/li2s8_gfn2_alpb_ether.xyz`) to the site.

RULE: tier 0 only ranks and filters. No number from here enters the paper.
Every row carries `tier0.method` (level of theory) and `tier0.wall_s`.

Written limitations: GFN2-xTB is not DFT; ALPB binding energies are relative
and serve for ranking (calibration against the Joule 2021 table at H3), not
as absolute values. The site search is local (no global docking) and can miss
chelation modes of long glymes.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from pace_s import DATA_DIR
from pace_s.config import load_campaign
from pace_s.descriptors.geometry import (
    Structure,
    coordination_sites,
    embed_lowest,
    mol_with_hs,
    place_cluster,
    place_ion,
)
from pace_s.descriptors.rdkit_descriptors import rdkit_descriptors
from pace_s.descriptors.xtb import EH_TO_EV, EH_TO_KCAL, XtbError, find_crest, find_xtb, run_crest, run_xtb
from pace_s.generate.filters import canonical
from pace_s.generate.table import read_table

log = logging.getLogger("pace_s.tier0")

LI2S8_XYZ = DATA_DIR / "reference" / "li2s8_gfn2_alpb_ether.xyz"
LI2S8_ANCHOR = 0  # index of the Li anchored to the site (first atom of the xyz)

TIER0_COLUMNS = [
    "tier0.method", "tier0.homo_eV", "tier0.lumo_eV", "tier0.gap_eV",
    "tier0.esp_max_kcal_mol", "tier0.esp_min_kcal_mol",
    "tier0.li_binding_eV", "tier0.li2s8_binding_eV", "tier0.solv_energy_kcal_mol",
    "tier0.dipole_D", "tier0.volume_A3", "tier0.tpsa_A2", "tier0.logp", "tier0.mol_weight",
    "tier0.frac_fluorination", "tier0.ratio_C_O", "tier0.n_conformers",
    "tier0.n_li_sites", "tier0.wall_s", "tier0.error",
]


class Tier0Settings:
    def __init__(self, cfg: dict[str, Any], threads: int | None = None, keep_files: bool = False):
        t = cfg.get("tiers", {}).get("tier0", {})
        self.gfn = 2 if str(t.get("method", "GFN2-xTB")).upper().startswith("GFN2") else 1
        self.alpb = t.get("alpb_solvent", "ether") if str(t.get("solvation", "ALPB")).upper() == "ALPB" else None
        self.conformers = str(t.get("conformers", "rdkit")).lower()
        self.n_conformers = int(t.get("n_rdkit_conformers", 10))
        self.max_sites = int(t.get("max_li_sites", 4))
        self.timeout_s = int(t.get("max_cpu_min_per_molecule", 5)) * 60 * 3  # slack over the target
        self.threads = threads
        self.keep_files = keep_files
        self.method = f"GFN{self.gfn}-xTB" + (f"/ALPB({self.alpb})" if self.alpb else "/gas")


# ----------------------------------------------------------------------------
# references computed once per working directory
# ----------------------------------------------------------------------------


@lru_cache(maxsize=None)
def reference_energies(workdir: str, gfn: int, alpb: str | None, threads: int | None) -> dict[str, float]:
    """E(Li+) and E(Li2S8) at the same level of theory as the molecules."""
    ref = Path(workdir) / "_reference"
    ref.mkdir(parents=True, exist_ok=True)
    li = Structure(["Li"], np.zeros((1, 3)))
    li_xyz = li.write_xyz(ref / "li.xyz", "Li+")
    e_li = run_xtb(li_xyz, ref / "li", charge=1, opt=False, gfn=gfn, alpb=alpb, threads=threads).energy_eh
    e_ps = run_xtb(LI2S8_XYZ, ref / "li2s8", charge=0, opt=True, gfn=gfn, alpb=alpb, threads=threads).energy_eh
    return {"Li+": e_li, "Li2S8": e_ps}


def li2s8_structure(workdir: str) -> Structure:
    opt = Path(workdir) / "_reference" / "li2s8" / "xtbopt.xyz"
    return Structure.from_xyz(opt if opt.exists() else LI2S8_XYZ)


# ----------------------------------------------------------------------------
# per molecule
# ----------------------------------------------------------------------------


def _binding_scan(
    struct: Structure,
    mol: Chem.Mol,
    sites: list[int],
    build,
    charge: int,
    e_mol: float,
    e_ref: float,
    workdir: Path,
    st: Tier0Settings,
) -> tuple[float | None, int]:
    """Minimum over sites of E(complex) - E(mol) - E(ref), in eV. Returns (E_bind, n_ok)."""
    best, n_ok = None, 0
    workdir.mkdir(parents=True, exist_ok=True)
    for k, site in enumerate(sites):
        d = workdir / f"site{k}"
        try:
            xyz = build(struct, mol, site).write_xyz(workdir / f"site{k}.xyz")
            res = run_xtb(xyz, d, charge=charge, opt=True, gfn=st.gfn, alpb=st.alpb, threads=st.threads,
                          timeout_s=st.timeout_s)
        except (XtbError, OSError, ValueError) as e:
            log.debug("site %d failed: %s", site, e)
            continue
        e_bind = (res.energy_eh - e_mol - e_ref) * EH_TO_EV
        n_ok += 1
        if best is None or e_bind < best:
            best = e_bind
    return best, n_ok


def compute_tier0(smiles: str, workdir: str | Path, st: Tier0Settings) -> dict[str, Any]:
    t0 = time.time()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"tier0.method": st.method}
    try:
        canon = canonical(smiles)
        if canon is None:
            raise ValueError("invalid SMILES")
        for k, v in rdkit_descriptors(canon).items():
            out[f"tier0.{k}"] = v

        # 2. conformer
        mol, cid, n_conf = embed_lowest(mol_with_hs(canon), st.n_conformers)
        out["tier0.n_conformers"] = n_conf
        out["tier0.volume_A3"] = float(AllChem.ComputeMolVolume(mol, confId=cid))
        struct = Structure.from_mol(mol, cid)
        xyz = struct.write_xyz(workdir / "rdkit.xyz", canon)
        if st.conformers == "crest" and find_crest():
            xyz = run_crest(xyz, workdir / "crest", gfn=st.gfn, alpb=st.alpb, threads=st.threads)
            out["tier0.method"] = st.method + "+CREST"

        # 3. optimization in implicit solvent
        res = run_xtb(xyz, workdir / "opt", charge=0, opt=True, gfn=st.gfn, alpb=st.alpb, threads=st.threads,
                      timeout_s=st.timeout_s)
        out.update({
            "tier0.homo_eV": res.homo_eV, "tier0.lumo_eV": res.lumo_eV, "tier0.gap_eV": res.gap_eV,
            "tier0.dipole_D": res.dipole_D,
            "tier0.solv_energy_kcal_mol": None if res.gsolv_eh is None else res.gsolv_eh * EH_TO_KCAL,
        })
        struct = Structure.from_xyz(res.opt_xyz)
        e_mol = res.energy_eh

        # 4. surface ESP. A failure here must not prevent the binding scans below;
        #    the row is flagged in tier0.error but Li+ and Li2S8 are still computed.
        #    First attempt with the requested threads; on failure, one retry with a
        #    single thread (the Windows crash seems to be in the parallel ESP code).
        esp_error = None
        for attempt, (sub, threads) in enumerate((("esp", st.threads), ("esp_retry", 1))):
            try:
                esp = run_xtb(res.opt_xyz, workdir / sub, charge=0, opt=False, gfn=st.gfn, alpb=st.alpb,
                              esp=True, threads=threads, timeout_s=st.timeout_s)
                out["tier0.esp_min_kcal_mol"] = esp.esp_min_kcal_mol
                out["tier0.esp_max_kcal_mol"] = esp.esp_max_kcal_mol
                esp_error = None
                if attempt:
                    log.info("%s: ESP recovered on the single-thread retry", smiles)
                break
            except XtbError as e:
                esp_error = f"esp: {str(e).splitlines()[0][:120]}"
        if esp_error:
            log.warning("%s: %s (after retry)", smiles, esp_error)

        # 5 and 6. binding with Li+ and with Li2S8
        refs = reference_energies(str(workdir.parent), st.gfn, st.alpb, st.threads)
        sites = coordination_sites(mol, res.charges, st.max_sites)
        out["tier0.n_li_sites"] = len(sites)
        if sites:
            e_li, _ = _binding_scan(struct, mol, sites, place_ion, 1, e_mol, refs["Li+"], workdir / "li", st)
            out["tier0.li_binding_eV"] = e_li
            cluster = li2s8_structure(str(workdir.parent))
            build = lambda s, m, i: place_cluster(s, m, i, cluster, LI2S8_ANCHOR)  # noqa: E731
            e_ps, _ = _binding_scan(struct, mol, sites, build, 0, e_mol, refs["Li2S8"], workdir / "li2s8", st)
            out["tier0.li2s8_binding_eV"] = e_ps
        out["tier0.error"] = esp_error
    except Exception as e:  # noqa: BLE001 - one failed molecule must not kill the block
        log.warning("%s: %s", smiles, e)
        out["tier0.error"] = f"{type(e).__name__}: {str(e)[:200]}"
    out["tier0.wall_s"] = round(time.time() - t0, 1)
    if not st.keep_files:
        shutil.rmtree(workdir, ignore_errors=True)
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def run_block(items: list[tuple[str, str]], workdir: Path, st: Tier0Settings) -> pd.DataFrame:
    rows = []
    for i, (cid, smi) in enumerate(items, 1):
        log.info("[%d/%d] %s %s", i, len(items), cid, smi)
        row = {"candidate_id": cid, "canonical_smiles": canonical(smi) or smi}
        row.update(compute_tier0(smi, workdir / cid, st))
        log.info("    %.1f s  Li+ %s eV  Li2S8 %s eV  gap %s eV",
                 row["tier0.wall_s"], _fmt(row.get("tier0.li_binding_eV")),
                 _fmt(row.get("tier0.li2s8_binding_eV")), _fmt(row.get("tier0.gap_eV")))
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in TIER0_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[["candidate_id", "canonical_smiles"] + TIER0_COLUMNS]


def _fmt(v: Any) -> str:
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.2f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="CSV with candidate_id + tier0.* columns")
    ap.add_argument("--workdir", required=True, help="calculation directory (one per campaign)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--candidates", help="candidate table (CSV/parquet) from the generator")
    src.add_argument("--smiles", nargs="+", help="loose SMILES, for testing")
    ap.add_argument("--start", type=int, default=0, help="first row of the block (job array)")
    ap.add_argument("--n", type=int, default=None, help="block size")
    ap.add_argument("--only-passing", action="store_true", default=True,
                    help="only rows with status=generated (default)")
    ap.add_argument("--threads", type=int, default=None, help="OpenMP threads for xtb (default: OMP_NUM_THREADS or 1)")
    ap.add_argument("--keep-files", action="store_true", help="keep the xtb working directories")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_campaign(args.config)
    st = Tier0Settings(cfg, threads=args.threads, keep_files=args.keep_files)

    if args.smiles:
        items = [(f"ADHOC-{i:04d}", s) for i, s in enumerate(args.smiles)]
    else:
        df = read_table(args.candidates)
        if args.only_passing and "status" in df.columns:
            df = df[df["status"] == "generated"]
        df = df.iloc[args.start : (args.start + args.n) if args.n else None]
        items = list(zip(df["candidate_id"], df["canonical_smiles"]))
    if not items:
        log.warning("empty block (start=%d)", args.start)
        return 0

    log.info("tier 0 (%s) on %d molecules; xtb=%s", st.method, len(items), find_xtb())
    table = run_block(items, Path(args.workdir), st)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    n_err = int(table["tier0.error"].notna().sum())
    log.info("wrote %s: %d rows, %d with errors, %.0f s total",
             out, len(table), n_err, table["tier0.wall_s"].sum())
    return 0


if __name__ == "__main__":
    sys.exit(main())
