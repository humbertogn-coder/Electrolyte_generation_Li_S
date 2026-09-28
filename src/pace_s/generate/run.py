"""Generate the stage-1 design space from the seeds.

    python -m pace_s.generate.run --config workflows/configs/campaign_LiS.yaml \
        --out data/campaigns/AL-01/candidates_stage1.csv [--rounds 2] [--max-size 60000]

Breadth-first search by rounds: in each round every operator listed in
`generation.stage1_operators` is applied to the molecules that passed the
hard filter in the previous round. Every new child is recorded with its
parent and operator, filtered (`filters.hard_filter` and the synthesizability
filter) and given its distance to the seeds. Nothing is dropped silently:
molecules that fail stay in the table with `status = filtered_out` and the reason.

The output satisfies the flattened `contracts/candidate_table.schema.json`
(`generate.table.validate_frame` checks it before writing).
"""

from __future__ import annotations

import argparse
import logging
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from pace_s import REPO_ROOT
from pace_s.config import load_campaign
from pace_s.generate import operators as ops
from pace_s.generate.filters import canonical, hard_filter, inchikey
from pace_s.generate.seeds import Seed, load_seeds
from pace_s.generate.synth import passes_synthesizability, rascore, sascore
from pace_s.generate.table import SCHEMA_VERSION, candidate_id, flatten, validate_frame

log = logging.getLogger("pace_s.generate")

_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

CHILD_ROLE = {"seed_solvent": "solvent", "seed_diluent": "diluent"}


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip() or "0000000"
    except Exception:  # noqa: BLE001
        return "0000000"


class SeedDistance:
    """1 - maximum Tanimoto similarity (Morgan r=2, 2048 bits) to the seeds."""

    def __init__(self, seeds: list[Seed]):
        self._fps = [_FPGEN.GetFingerprint(Chem.MolFromSmiles(s.smiles)) for s in seeds]

    def __call__(self, smiles: str) -> float:
        fp = _FPGEN.GetFingerprint(Chem.MolFromSmiles(smiles))
        sims = DataStructs.BulkTanimotoSimilarity(fp, self._fps)
        return round(1.0 - max(sims), 4)


def _record(
    n: int,
    smiles: str,
    role: str,
    stage: int,
    parent_id: str | None,
    operator: str | None,
    cfg: dict[str, Any],
    dist: SeedDistance,
    commit: str,
) -> dict[str, Any]:
    canon = canonical(smiles)
    assert canon is not None
    ds = cfg["design_space"]
    hard = hard_filter(canon, ds)
    sa = sascore(canon)
    ra = rascore(canon)
    synth_ok = passes_synthesizability(sa, ra, ds.get("synthesizability", {}))

    if not hard.passes:
        status, reason = "filtered_out", hard.reason
    elif not synth_ok:
        status, reason = "filtered_out", f"synthesizability:sascore={sa:.2f}" if sa is not None else "synthesizability"
    else:
        status, reason = "generated", None

    filters: dict[str, Any] = {"passes_hard": hard.passes and synth_ok, "hard_fail_reason": reason}
    if sa is not None:
        filters["sascore"] = round(sa, 3)
    if ra is not None:
        filters["rascore"] = round(ra, 3)

    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id(n),
        "smiles": smiles,
        "canonical_smiles": canon,
        "inchikey": inchikey(canon),
        "role": role,
        "generation": {
            "stage": stage,
            "parent_id": parent_id,
            "operator": operator,
            "distance_to_seeds": dist(canon),
        },
        "filters": filters,
        "status": status,
        "campaign_id": cfg["campaign_id"],
        "commit": commit,
    }


def generate_space(
    cfg: dict[str, Any],
    seeds: list[Seed] | None = None,
    n_rounds: int | None = None,
    max_size: int | None = None,
) -> pd.DataFrame:
    gen_cfg = cfg["generation"]
    seeds = seeds if seeds is not None else load_seeds()
    n_rounds = n_rounds if n_rounds is not None else int(gen_cfg.get("n_rounds", 2))
    max_size = max_size if max_size is not None else int(gen_cfg["target_size"]["max"])
    operator_names = list(gen_cfg["stage1_operators"])
    unknown = set(operator_names) - set(ops.OPERATORS)
    if unknown:
        raise ValueError(f"Unknown operators in the configuration: {sorted(unknown)}")

    dist = SeedDistance(seeds)
    commit = git_commit()
    records: list[dict[str, Any]] = []
    seen: dict[str, str] = {}  # canonical -> candidate_id
    n = 0

    # round 0: seeds
    for s in seeds:
        n += 1
        rec = _record(n, s.smiles, s.role, 0, None, "seed", cfg, dist, commit)
        seen[rec["canonical_smiles"]] = rec["candidate_id"]
        records.append(rec)
    frontier = [r for r in records if r["status"] == "generated"]
    log.info("round 0: %d seeds, %d pass the filter", len(seeds), len(frontier))

    per_operator: Counter = Counter()
    rng = random.Random(int(cfg.get("seed", 0)))
    children_per_parent = float(gen_cfg.get("expected_children_per_parent", 25.0))
    for rnd in range(1, n_rounds + 1):
        t0 = time.time()
        next_frontier: list[dict[str, Any]] = []
        n_new = 0
        # If the full round would exceed max_size, the frontier is randomly
        # subsampled (fixed seed) instead of truncated by order: coverage of the
        # space stays uniform and reproducible, not alphabetical.
        remaining = max_size - len(records)
        expected = len(frontier) * children_per_parent
        if expected > remaining and len(frontier) > 1:
            k = max(1, int(remaining / children_per_parent))
            frontier = rng.sample(sorted(frontier, key=lambda r: r["canonical_smiles"]), k)
            log.info("round %d: frontier subsampled to %d parents to respect max_size=%d", rnd, k, max_size)
        for parent in sorted(frontier, key=lambda r: r["canonical_smiles"]):
            role = CHILD_ROLE.get(parent["role"], parent["role"])
            for name in operator_names:
                for child in sorted(ops.OPERATORS[name](parent["canonical_smiles"])):
                    if child in seen:
                        continue
                    n += 1
                    rec = _record(n, child, role, 1, parent["candidate_id"], name, cfg, dist, commit)
                    seen[child] = rec["candidate_id"]
                    records.append(rec)
                    per_operator[name] += 1
                    n_new += 1
                    if rec["status"] == "generated":
                        next_frontier.append(rec)
                    if len(records) >= max_size:
                        break
                if len(records) >= max_size:
                    break
            if len(records) >= max_size:
                break
        n_ok = len(next_frontier)
        if frontier:
            children_per_parent = max(1.0, n_new / len(frontier))
        log.info("round %d: %d new, %d pass the filters, total %d (%.1f s)",
                 rnd, n_new, n_ok, len(records), time.time() - t0)
        frontier = next_frontier
        if len(records) >= max_size:
            log.info("reached max_size=%d", max_size)
            break
        if not frontier:
            break

    df = pd.DataFrame([flatten(r) for r in records])
    log.info("per operator: %s", dict(per_operator))
    log.info("per status: %s", df["status"].value_counts().to_dict())
    fails = df.loc[df["status"] == "filtered_out", "filters.hard_fail_reason"]
    log.info("rejection reasons: %s", fails.str.split(":").str[0].value_counts().to_dict())
    return df


def write_table(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help=".csv or .parquet")
    ap.add_argument("--rounds", type=int, default=None, help="mutation rounds (default: generation.n_rounds)")
    ap.add_argument("--max-size", type=int, default=None, help="default: generation.target_size.max")
    ap.add_argument("--no-validate", action="store_true", help="skip row validation against the contract")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_campaign(args.config)
    df = generate_space(cfg, n_rounds=args.rounds, max_size=args.max_size)

    if not args.no_validate:
        msgs = validate_frame(df)
        if msgs:
            for m in msgs:
                log.error(m)
            log.error("the table does not satisfy the candidate_table contract; nothing written")
            return 1
        log.info("all rows validate against candidate_table.schema.json")

    out = Path(args.out)
    write_table(df, out)
    log.info("wrote %s (%d rows)", out, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
