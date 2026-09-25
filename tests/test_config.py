"""Cambiar de Li-S a Li-SPAN es cambiar el YAML, no el código."""

import pytest

from pace_s.config import OBJECTIVE_KEYS, dominant_objective, load_campaign, objective_weights


def test_objectives_complete(cfg_lis, cfg_lispan):
    assert set(cfg_lis["objectives"]) == set(OBJECTIVE_KEYS)
    assert set(cfg_lispan["objectives"]) == set(OBJECTIVE_KEYS)


def test_dominant_objectives(cfg_lis, cfg_lispan):
    assert dominant_objective(cfg_lis) == "O2_polysulfide_retention"
    assert dominant_objective(cfg_lispan) == "O3_reductive_passivation"


def test_o5_is_constraint(cfg_lis, cfg_lispan):
    assert "O5_synthesizability" not in objective_weights(cfg_lis)
    assert "O5_synthesizability" not in objective_weights(cfg_lispan)


def test_lispan_inherits_design_space(cfg_lis, cfg_lispan):
    assert cfg_lispan["design_space"] == cfg_lis["design_space"]
    assert cfg_lispan["tiers"] == cfg_lis["tiers"]
    assert cfg_lispan["system"] == "Li-SPAN"
    assert cfg_lispan["campaign_id"] != cfg_lis["campaign_id"]


def test_gate_threshold(cfg_lis):
    assert cfg_lis["calibration"]["gate"]["threshold"] == 0.6
    assert cfg_lis["active_learning"]["random_baseline"] is True


def test_salts_fixed(cfg_lis):
    assert cfg_lis["salts"] == ["LiFSI", "LiTFSI"]


def test_bad_system(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("campaign_id: X-01\nsystem: Li-ion\nobjectives: {}\ndesign_space: {}\n")
    with pytest.raises(ValueError):
        load_campaign(p)
