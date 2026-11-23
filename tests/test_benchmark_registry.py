"""Benchmark registry rules (MASTER_PLAN §5.5-5.7, §10.1, §12.2)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from royale_ai.benchmarks import registry
from royale_ai.opponents.registry import PILOT_TIERS, PILOTS, PilotUnavailable, get_pilot

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _doc() -> dict:
    return yaml.safe_load((CONFIGS / "benchmarks.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def decks():
    return registry.load_deck_configs(CONFIGS)


def test_registry_loads_and_has_both_g2_baselines() -> None:
    b = registry.load_benchmarks(CONFIGS)
    assert {"hog_mirror_random_legal", "hog_mirror_noop"} <= set(b)
    assert b["hog_mirror_random_legal"].pilot == "random_legal"
    assert b["hog_mirror_noop"].pilot == "noop"
    assert b["hog_mirror_random_legal"].id != b["hog_mirror_noop"].id


def test_all_plan_benchmark_ids_registered() -> None:
    ids = set(registry.load_benchmarks(CONFIGS))
    assert ids == {
        "engine_legal_contract", "hog_mirror_random_legal", "hog_mirror_noop",
        "hog_mirror_scripted", "hog_vs_giant_scripted", "hog_vs_control_scripted",
        "hog_tactical_scenarios", "hog_vs_frozen_checkpoint", "champion_paired_seat_swap",
        "multideck_matchup_matrix", "heldout_deck_zero_shot", "unseen_card_transfer",
    }


def test_missing_g2_baseline_rejected(decks) -> None:
    doc = _doc()
    doc["benchmarks"] = [b for b in doc["benchmarks"] if b["id"] != "hog_mirror_noop"]
    with pytest.raises(registry.RegistryError, match="hog_mirror_noop"):
        registry.validate_benchmarks(doc, decks)


def test_unregistered_pilot_rejected(decks) -> None:
    doc = _doc()
    doc["benchmarks"][1]["pilot"] = "scripted_hog"  # retired id
    with pytest.raises(KeyError, match="retired"):
        registry.validate_benchmarks(doc, decks)
    doc["benchmarks"][1]["pilot"] = "my_bot"
    with pytest.raises(KeyError, match="unregistered"):
        registry.validate_benchmarks(doc, decks)


def test_unregistered_deck_rejected(decks) -> None:
    doc = _doc()
    doc["benchmarks"][1]["opponent_deck"] = "random"
    with pytest.raises(registry.RegistryError, match="unregistered"):
        registry.validate_benchmarks(doc, decks)


def test_hog_pilot_never_pilots_giant(decks) -> None:
    doc = _doc()
    giant = next(b for b in doc["benchmarks"] if b["id"] == "hog_vs_giant_scripted")
    giant["pilot"] = "hog_cycle"
    with pytest.raises(registry.RegistryError, match="Hog pilot"):
        registry.validate_benchmarks(doc, decks)


def test_deck_c_pilot_is_pinned(decks) -> None:
    doc = _doc()
    ctl = next(b for b in doc["benchmarks"] if b["id"] == "hog_vs_control_scripted")
    ctl["pilot"] = "patient"
    with pytest.raises(registry.RegistryError, match="pinned"):
        registry.validate_benchmarks(doc, decks)


def test_cross_deck_benchmark_cannot_be_active_before_admission(decks) -> None:
    doc = _doc()
    giant = next(b for b in doc["benchmarks"] if b["id"] == "hog_vs_giant_scripted")
    giant["status"] = "active"
    with pytest.raises(registry.RegistryError, match="pending"):
        registry.validate_benchmarks(doc, decks)


def test_explicit_blue_red_seat_pairs_required(decks) -> None:
    doc = copy.deepcopy(_doc())
    doc["benchmarks"][1]["agent_seats"] = ["blue"]
    with pytest.raises(registry.RegistryError, match="blue, red"):
        registry.validate_benchmarks(doc, decks)
    doc["benchmarks"][1]["agent_seats"] = [0, 1]
    with pytest.raises(registry.RegistryError):
        registry.validate_benchmarks(doc, decks)


def test_pilots_have_tiers_and_unimplemented_ones_refuse() -> None:
    for p in PILOTS.values():
        assert p.tier in PILOT_TIERS
    assert get_pilot("noop").tier == "passive"
    assert get_pilot("random_legal").tier == "random"
    with pytest.raises(PilotUnavailable):
        get_pilot("hog_cycle").build()
    assert get_pilot("frozen_hog_v3").tier == "frozen_learned"


def test_scenario_ids_match_plan() -> None:
    scen = registry.load_scenarios(CONFIGS)
    assert tuple(s["id"] for s in scen) == registry.SCENARIO_IDS
    giant = next(s for s in scen if s["id"] == "giant_support_separation")
    assert "starter_giant" in giant["requires"]
