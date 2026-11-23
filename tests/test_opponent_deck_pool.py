"""Opponent decks and pilots (MASTER_PLAN §5.3-5.7). P0 checks the definitions; G1 admits."""

from __future__ import annotations

import pytest
import royalegym as rg
from royalegym import RustEngine

from royale_ai.benchmarks.registry import load_deck_configs
from royale_ai.decks import DECKS, HOG_26, HOG_CONTROL, STARTER_GIANT
from royale_ai.opponents.registry import PILOTS


@pytest.fixture(scope="module")
def catalogue() -> set[str]:
    return {c.name for c in RustEngine().cards()}


def test_every_deck_card_exists_in_engine(catalogue) -> None:
    for key, cards in DECKS.items():
        assert set(cards) <= catalogue, (key, set(cards) - catalogue)


def test_deck_b_new_cards_are_exactly_the_g1_audit_set() -> None:
    cfg = load_deck_configs()
    assert set(STARTER_GIANT) - set(HOG_26) == {"Knight", "Archer", "Giant", "Minions", "Zap"}
    assert set(cfg["starter_giant"]["new_cards_to_audit"]) == set(STARTER_GIANT) - set(HOG_26)
    assert set(STARTER_GIANT) == set(rg.STARTER_DECK)  # the simulator's documented starter deck


def test_deck_c_only_new_card_is_tesla_and_pilot_pinned() -> None:
    cfg = load_deck_configs()
    assert set(HOG_CONTROL) - set(HOG_26) - set(STARTER_GIANT) == {"Tesla"}
    assert cfg["hog_control"]["pinned_pilot"] == "first_affordable"
    assert cfg["hog_control"]["admission"]["status"].startswith("pending")


def test_admission_states() -> None:
    cfg = load_deck_configs()
    assert cfg["hog_26"]["admission"]["gate"] == "G0"
    assert cfg["starter_giant"]["admission"]["status"] == "pending_g1"


def test_giant_pilots_are_weak_builtin_scripted() -> None:
    cfg = load_deck_configs()
    for p in cfg["starter_giant"]["eligible_pilots"]:
        assert not PILOTS[p].hog_only and PILOTS[p].factory is not None


@pytest.mark.parametrize("deck_key", ["hog_26", "starter_giant", "hog_control"])
@pytest.mark.parametrize("agent_seat", ["blue", "red"])
def test_every_deck_runs_a_full_match_on_both_seats(deck_key: str, agent_seat: str) -> None:
    """Both-seat entry points: the deck completes a deterministic match as either seat
    (admission rule 4). Loading-only is not admission; G1 audits the mechanics."""
    deck = list(DECKS[deck_key])
    pair = [deck, list(HOG_26)] if agent_seat == "blue" else [list(HOG_26), deck]
    env = rg.make_env(deck=pair, engine="rust")
    b1 = rg.play_battle(env, "first-affordable", "random", seed=3)
    b2 = rg.play_battle(env, "first-affordable", "random", seed=3)
    assert b1.winner is not None and b1 == b2
