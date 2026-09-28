"""Merge tier-0 block outputs back into the candidate table.

    python -m pace_s.descriptors.merge_tier0 \
        --candidates data/campaigns/AL-01/candidates_stage1.csv \
        --blocks data/campaigns/AL-01/tier0 \
        --out data/campaigns/AL-01/candidates_tier0.csv

The job array (`workflows/slurm/tier0_xtb_array.sbatch`) writes one CSV per
block (`tier0_000000.csv`, `tier0_000050.csv`, ...). This script:

1. reads every block, concatenates them and deduplicates by `candidate_id`
   (last block wins, so a resubmitted block overrides the old one);
2. left-joins the `tier0.*` columns onto the candidate table;
3. sets `status = tier0_done` for rows that got a tier-0 result without error,
   leaves `generated` for the rest, and never touches `filtered_out`;
4. validates every row against `candidate_table.schema.json`;
5. reports coverage and the failure reasons, and writes `<out>.missing.csv`
   with the candidates that still need tier 0 (not computed, or computed with
   error), ready to be fed back to the job array with `--candidates`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from pace_s.descriptors.run_tier0 import TIER0_COLUMNS
from pace_s.generate.table import STRING_COLUMNS, read_table, validate_frame

log = logging.getLogger("pace_s.merge_tier0")


def read_blocks(blocks_dir: str | Path, pattern: str = "tier0_*.csv") -> pd.DataFrame:
    """Concatenate all block CSVs; on duplicate candidate_id the last file (sorted by name) wins."""
    files = sorted(Path(blocks_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no block files matching {pattern!r} in {blocks_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype={c: str for c in STRING_COLUMNS})
        if "candidate_id" not in df.columns:
            log.warning("skipping %s: no candidate_id column", f.name)
            continue
        df["tier0.block_file"] = f.name
        frames.append(df)
    blocks = pd.concat(frames, ignore_index=True)
    n_dup = int(blocks["candidate_id"].duplicated().sum())
    if n_dup:
        log.info("%d duplicated candidate_id across blocks; keeping the last occurrence", n_dup)
    blocks = blocks.drop_duplicates("candidate_id", keep="last")
    log.info("read %d block files, %d unique candidates", len(files), len(blocks))
    return blocks


def merge(candidates: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """Join tier-0 columns onto the candidate table and update `status`."""
    tier0 = blocks.reindex(columns=["candidate_id"] + TIER0_COLUMNS)  # missing columns become NaN
    # drop any stale tier0.* columns from a previous merge before joining
    stale = [c for c in candidates.columns if c.startswith("tier0.")]
    merged = candidates.drop(columns=stale).merge(tier0, on="candidate_id", how="left")

    has_result = merged["tier0.method"].notna()
    failed = has_result & merged["tier0.error"].notna()
    done = has_result & ~failed & (merged["status"] != "filtered_out")
    merged.loc[done, "status"] = "tier0_done"
    # a candidate whose tier-0 run failed keeps its previous status; the error text stays in tier0.error
    return merged


def missing_candidates(merged: pd.DataFrame) -> pd.DataFrame:
    """Candidates that passed the filters but have no valid tier-0 result yet."""
    pending = (merged["status"] == "generated")
    return merged.loc[pending, ["candidate_id", "canonical_smiles", "status", "tier0.error"]]


def summarize(merged: pd.DataFrame) -> dict:
    eligible = merged["status"].isin(["generated", "tier0_done"])
    done = merged["status"] == "tier0_done"
    errors = merged["tier0.error"].dropna()
    out = {
        "rows": int(len(merged)),
        "eligible": int(eligible.sum()),
        "tier0_done": int(done.sum()),
        "pending": int((merged["status"] == "generated").sum()),
        "failed": int(len(errors)),
        "coverage": round(float(done.sum() / max(eligible.sum(), 1)), 4),
    }
    if len(errors):
        out["error_types"] = errors.str.split(":").str[0].value_counts().to_dict()
    w = merged.loc[done, "tier0.wall_s"].dropna()
    if len(w):
        out["wall_s_median"] = round(float(w.median()), 1)
        out["wall_s_total_h"] = round(float(w.sum() / 3600), 2)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, help="candidate table from the generator (CSV/parquet)")
    ap.add_argument("--blocks", required=True, help="directory with the tier0_*.csv block files")
    ap.add_argument("--out", required=True, help="merged table (.csv or .parquet)")
    ap.add_argument("--pattern", default="tier0_*.csv")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    candidates = read_table(args.candidates)
    blocks = read_blocks(args.blocks, args.pattern)
    unknown = set(blocks["candidate_id"]) - set(candidates["candidate_id"])
    if unknown:
        log.warning("%d block rows have candidate_id not in the table (ignored): %s",
                    len(unknown), sorted(unknown)[:5])
    merged = merge(candidates, blocks)

    if not args.no_validate:
        msgs = validate_frame(merged)
        if msgs:
            for m in msgs:
                log.error(m)
            log.error("merged table does not satisfy the candidate_table contract; nothing written")
            return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        merged.to_parquet(out, index=False)
    else:
        merged.to_csv(out, index=False)
    missing = missing_candidates(merged)
    missing_path = out.with_suffix(".missing.csv")
    missing.to_csv(missing_path, index=False)

    s = summarize(merged)
    log.info("summary: %s", s)
    log.info("wrote %s (%d rows) and %s (%d pending)", out, len(merged), missing_path, len(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
