"""Merging tier-0 blocks back into the candidate table (no xtb needed:
the blocks are synthetic)."""

import numpy as np
import pandas as pd
import pytest

from pace_s.descriptors.merge_tier0 import main, merge, missing_candidates, read_blocks, summarize
from pace_s.descriptors.run_tier0 import TIER0_COLUMNS
from pace_s.generate.run import generate_space
from pace_s.generate.seeds import Seed
from pace_s.generate.table import validate_frame

SEEDS = [
    Seed("DME", "seed_solvent", "COCCOC", "1,2-dimethoxyethane"),
    Seed("TTE", "seed_diluent", "FC(F)C(F)(F)OCC(F)(F)C(F)F", "TTE"),
]


@pytest.fixture(scope="module")
def candidates(cfg_lis):
    return generate_space(cfg_lis, seeds=SEEDS, n_rounds=1, max_size=1000)


def _fake_block(ids, error=None, method="GFN2-xTB/ALPB(ether)", wall=7.0):
    rows = []
    for cid in ids:
        row = {"candidate_id": cid, "canonical_smiles": "X"}
        row.update({c: np.nan for c in TIER0_COLUMNS})
        row.update({
            "tier0.method": method, "tier0.homo_eV": -11.0, "tier0.lumo_eV": -0.1, "tier0.gap_eV": 10.9,
            "tier0.esp_max_kcal_mol": 18.0, "tier0.esp_min_kcal_mol": -45.0,
            "tier0.li_binding_eV": -2.6, "tier0.li2s8_binding_eV": -1.5, "tier0.solv_energy_kcal_mol": -6.0,
            "tier0.dipole_D": 2.3, "tier0.volume_A3": 97.0, "tier0.tpsa_A2": 18.5, "tier0.logp": 0.3,
            "tier0.mol_weight": 90.1, "tier0.frac_fluorination": 0.0, "tier0.ratio_C_O": 2.0,
            "tier0.n_conformers": 7, "tier0.n_li_sites": 2, "tier0.wall_s": wall, "tier0.error": error,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def test_merge_sets_status_and_keeps_filtered(candidates):
    ok = candidates[candidates["status"] == "generated"]["candidate_id"].tolist()
    filtered = candidates[candidates["status"] == "filtered_out"]["candidate_id"].tolist()
    assert len(ok) > 10 and len(filtered) > 0
    done_ids, err_ids, pending_ids = ok[:5], ok[5:7], ok[7:]
    blocks = pd.concat([
        _fake_block(done_ids),
        _fake_block(err_ids, error="XtbError: xtb did not terminate normally"),
        _fake_block(filtered[:1]),  # a filtered-out molecule computed by mistake stays filtered_out
    ], ignore_index=True)
    merged = merge(candidates, blocks)

    st = merged.set_index("candidate_id")["status"]
    assert (st.loc[done_ids] == "tier0_done").all()
    assert (st.loc[err_ids] == "generated").all()
    assert (st.loc[pending_ids] == "generated").all()
    assert (st.loc[filtered] == "filtered_out").all()
    assert merged.set_index("candidate_id").loc[err_ids, "tier0.error"].notna().all()
    assert len(merged) == len(candidates)
    assert validate_frame(merged) == []

    miss = missing_candidates(merged)
    assert set(miss["candidate_id"]) == set(err_ids) | set(pending_ids)

    s = summarize(merged)
    assert s["tier0_done"] == 5 and s["failed"] == 2 and s["pending"] == len(err_ids) + len(pending_ids)
    assert s["eligible"] == len(ok)
    assert s["error_types"] == {"XtbError": 2}
    assert s["wall_s_median"] == 7.0


def test_merge_is_idempotent_and_last_block_wins(candidates):
    ok = candidates[candidates["status"] == "generated"]["candidate_id"].tolist()[:3]
    first = merge(candidates, _fake_block(ok, wall=1.0))
    second = merge(first, _fake_block(ok, wall=2.0))  # re-merge over an already merged table
    assert (second.set_index("candidate_id").loc[ok, "tier0.wall_s"] == 2.0).all()
    assert list(second.columns) == list(first.columns)


def test_read_blocks_dedup(tmp_path, candidates):
    ok = candidates[candidates["status"] == "generated"]["candidate_id"].tolist()[:4]
    _fake_block(ok[:3], wall=1.0).to_csv(tmp_path / "tier0_000000.csv", index=False)
    _fake_block(ok[2:], wall=9.0).to_csv(tmp_path / "tier0_000003.csv", index=False)  # overlaps on ok[2]
    blocks = read_blocks(tmp_path)
    assert len(blocks) == 4
    assert float(blocks.set_index("candidate_id").loc[ok[2], "tier0.wall_s"]) == 9.0
    with pytest.raises(FileNotFoundError):
        read_blocks(tmp_path / "empty")


def test_cli_end_to_end(tmp_path, candidates):
    cand_path = tmp_path / "candidates.csv"
    candidates.to_csv(cand_path, index=False)
    blocks_dir = tmp_path / "tier0"
    blocks_dir.mkdir()
    ok = candidates[candidates["status"] == "generated"]["candidate_id"].tolist()
    _fake_block(ok[:6]).to_csv(blocks_dir / "tier0_000000.csv", index=False)
    out = tmp_path / "merged.csv"
    assert main(["--candidates", str(cand_path), "--blocks", str(blocks_dir), "--out", str(out)]) == 0
    merged = pd.read_csv(out)
    assert (merged["status"] == "tier0_done").sum() == 6
    missing = pd.read_csv(out.with_suffix(".missing.csv"))
    assert len(missing) == len(ok) - 6
    assert "tier0.block_file" not in merged.columns
