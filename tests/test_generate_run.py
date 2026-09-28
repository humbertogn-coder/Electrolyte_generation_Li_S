"""The generator produces a table that satisfies the contract, with provenance per row."""

import pandas as pd
import pytest

from pace_s.generate.run import generate_space, main
from pace_s.generate.seeds import Seed
from pace_s.generate.table import flatten, unflatten, validate_frame

SEEDS = [
    Seed("DME", "seed_solvent", "COCCOC", "1,2-dimethoxyethane"),
    Seed("TTE", "seed_diluent", "FC(F)C(F)(F)OCC(F)(F)C(F)F", "TTE"),
]


@pytest.fixture(scope="module")
def table(cfg_lis):
    return generate_space(cfg_lis, seeds=SEEDS, n_rounds=1, max_size=10_000)


def test_seeds_first_and_stage0(table):
    head = table.head(2)
    assert list(head["candidate_id"]) == ["PSM-000001", "PSM-000002"]
    assert list(head["generation.stage"]) == [0, 0]
    assert list(head["generation.operator"]) == ["seed", "seed"]
    assert head["generation.parent_id"].isna().all()
    assert (head["generation.distance_to_seeds"] == 0.0).all()


def test_children_have_parent_operator_and_role(table):
    kids = table[table["generation.stage"] == 1]
    assert len(kids) > 50
    assert kids["generation.parent_id"].isin({"PSM-000001", "PSM-000002"}).all()
    assert set(kids["generation.operator"]) <= {
        "h_to_f", "chain_extend", "chain_contract", "heteroatom_swap",
        "ring_open", "ring_close", "bridge_insert", "branch",
    }
    assert set(kids["role"]) == {"solvent", "diluent"}
    assert (kids["generation.distance_to_seeds"] > 0).all()


def test_no_duplicates(table):
    assert table["canonical_smiles"].is_unique
    assert table["candidate_id"].is_unique


def test_filtered_rows_keep_reason(table):
    out = table[table["status"] == "filtered_out"]
    assert len(out) > 0
    assert out["filters.hard_fail_reason"].notna().all()
    ok = table[table["status"] == "generated"]
    assert ok["filters.passes_hard"].all()
    assert ok["filters.hard_fail_reason"].isna().all()


def test_provenance_columns(table, cfg_lis):
    assert (table["campaign_id"] == cfg_lis["campaign_id"]).all()
    assert table["commit"].str.fullmatch(r"[0-9a-f]{7,40}").all()
    assert (table["schema_version"] == "1.0.0").all()


def test_rows_validate_against_contract(table):
    assert validate_frame(table) == []


def test_flatten_roundtrip():
    rec = {"a": 1, "b": {"c": None, "d": {"e": 2.5}}, "generation": {"parent_id": None}}
    flat = flatten(rec)
    assert flat == {"a": 1, "b.c": None, "b.d.e": 2.5, "generation.parent_id": None}
    back = unflatten(flat)
    assert back == {"a": 1, "b": {"d": {"e": 2.5}}, "generation": {"parent_id": None}}


def test_cli_writes_csv(tmp_path):
    out = tmp_path / "t.csv"
    rc = main(["--config", "campaign_LiS.yaml", "--out", str(out), "--rounds", "1", "--max-size", "200"])
    assert rc == 0
    df = pd.read_csv(out)
    # the frontier is subsampled so as not to exceed max_size; it may land slightly below
    assert 100 < len(df) <= 200
    assert "canonical_smiles" in df.columns


def test_unknown_operator_rejected(cfg_lis):
    bad = dict(cfg_lis)
    bad["generation"] = dict(cfg_lis["generation"], stage1_operators=["h_to_f", "teleport"])
    with pytest.raises(ValueError):
        generate_space(bad, seeds=SEEDS, n_rounds=1)
