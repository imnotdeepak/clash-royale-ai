"""Run manifest completeness (MASTER_PLAN §11.1)."""

from __future__ import annotations

import pytest
import yaml
from royalegym import RustEngine

from royale_ai.adapters.action_schema import TILES_18X32_V1
from royale_ai.telemetry.manifest import MANIFEST_KEYS, build_manifest, write_manifest


def _m(**kw):
    base = dict(
        experiment_id="t", engine=RustEngine(), config={"a": 1}, agent_deck="hog_26",
        opponent_deck="hog_26", opponent_pilot="noop", pilot_strength_tier="passive",
        seat="red", seed=7,
    )
    base.update(kw)
    return build_manifest(**base)


def test_all_plan_keys_present_and_seat_lowercase(tmp_path) -> None:
    m = _m()
    assert set(MANIFEST_KEYS) <= set(m)
    assert m["seat"] == "red" and m["seat_mapping_verified"].startswith("team0_blue__slot2i_blue_v1")
    assert m["action_schema_version"] == "tiles_18x32_v1" and m["action_schema_hash"] == TILES_18X32_V1.hash
    assert m["simulator_version"] and m["royalegym_version"]
    assert isinstance(m["card_data_version"], dict) and "cards_json_fnv1a64" in m["card_data_version"]
    p = write_manifest(m, tmp_path / "m.yaml")
    assert yaml.safe_load(p.read_text(encoding="utf-8"))["seed"] == 7


def test_bad_seat_rejected() -> None:
    with pytest.raises(ValueError):
        _m(seat="Red")


def test_fixed_step_gamma_recorded_only_for_path_a() -> None:
    m = _m(gamma_per_second=0.998001, rollout_discount_mode="fixed_step_stock_maskableppo")
    assert m["gamma_step_fixed"] == pytest.approx(0.999)
    assert _m(gamma_per_second=0.998001, rollout_discount_mode="custom_per_transition")["gamma_step_fixed"] is None
