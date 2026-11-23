"""Deck-split leakage control (MASTER_PLAN §10.3) and frozen seed manifests (§13 Task 6)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from royale_ai.benchmarks import registry, seeds
from royale_ai.decks import DECKS, HOG_26, canonical_deck_id

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_canonical_id_is_order_free_and_distinct() -> None:
    assert canonical_deck_id(HOG_26) == canonical_deck_id(list(reversed(HOG_26)))
    assert len({canonical_deck_id(d) for d in DECKS.values()}) == len(DECKS)
    with pytest.raises(ValueError):
        canonical_deck_id(list(HOG_26[:7]) + [HOG_26[0]])


def test_shipped_split_manifest_is_valid() -> None:
    pools = registry.load_splits(CONFIGS)
    assert canonical_deck_id(HOG_26) in pools["train"]


def _doc() -> dict:
    return yaml.safe_load((CONFIGS / "deck_splits.yaml").read_text(encoding="utf-8"))


def test_deck_in_two_pools_is_rejected() -> None:
    doc = _doc()
    doc["pools"]["test"] = copy.deepcopy(doc["pools"]["train"])
    with pytest.raises(registry.RegistryError, match="both train and test"):
        registry.validate_splits(doc)


def test_reordered_duplicate_is_still_caught() -> None:
    # Same eight cards under another key would canonicalize to the same id: emulate by id.
    doc = _doc()
    doc["pools"]["validation"] = [{"key": "hog_26", "canonical_id": canonical_deck_id(HOG_26[::-1])}]
    with pytest.raises(registry.RegistryError):
        registry.validate_splits(doc)


def test_wrong_canonical_id_is_rejected() -> None:
    doc = _doc()
    doc["pools"]["train"][0]["canonical_id"] = "deck_000000000000"
    with pytest.raises(registry.RegistryError, match="does not match"):
        registry.validate_splits(doc)


def test_frozen_split_cannot_be_a_draft() -> None:
    doc = _doc()
    doc["frozen"] = True
    with pytest.raises(registry.RegistryError, match="non-draft"):
        registry.validate_splits(doc)


def test_heldout_decks_excluded_from_training_generators() -> None:
    pools = {"train": set(), "validation": set(), "test": {canonical_deck_id(DECKS["starter_giant"])}}
    assert not registry.allowed_in_training(list(DECKS["starter_giant"])[::-1], pools)
    assert registry.allowed_in_training(list(HOG_26), pools)


def test_seed_manifest_frozen_and_disjoint() -> None:
    m = seeds.load(CONFIGS / "seeds" / "hog_mirror_v1.yaml")
    assert len(m["validation"]) == 200 and len(m["test"]) == 500
    assert not set(m["validation"]) & set(m["test"])
    assert not any(seeds.is_train_seed(s) for s in m["validation"] + m["test"])
    # Stable: regenerating from the recorded generator seed gives the same manifest.
    regen = seeds.generate(m["manifest_id"], m["generator_seed"], 200, 500)
    assert regen["sha256"] == m["sha256"]


def test_altered_seed_manifest_is_rejected(tmp_path: Path) -> None:
    text = (CONFIGS / "seeds" / "hog_mirror_v1.yaml").read_text(encoding="utf-8")
    m = yaml.safe_load(text)
    m["test"][0] += 1
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(m), encoding="utf-8")
    with pytest.raises(ValueError, match="altered"):
        seeds.load(p)


def test_every_seed_is_paired_on_both_seats() -> None:
    m = seeds.load(CONFIGS / "seeds" / "hog_mirror_v1.yaml")
    games = seeds.paired_games(m, "test")
    assert len(games) == 2 * len(m["test"])
    by_seed: dict[int, set[str]] = {}
    for s, seat in games:
        by_seed.setdefault(s, set()).add(seat)
    assert all(v == {"blue", "red"} for v in by_seed.values())
