"""Fair information (MASTER_PLAN §6.2): no hidden-hand, next-card or deck leakage, both seats."""

from __future__ import annotations

import numpy as np
import pytest
from royalegym import ClashParallelEnv, DefaultStateMutator, RustEngine, SpatialObsBuilder
from royalegym.obs import Reveal
from royalegym.protocol import ShuffleMode

from royale_ai.decks import HOG_26, HOG_CONTROL, STARTER_GIANT


def _env(blue: list[str], red: list[str], reveal: Reveal | None = None) -> ClashParallelEnv:
    builder = SpatialObsBuilder(reveal=reveal) if reveal is not None else SpatialObsBuilder()
    return ClashParallelEnv(
        RustEngine(),
        obs_builder=builder,
        state_mutator=DefaultStateMutator(decks=[blue, red], shuffle=ShuffleMode.NONE),
    )


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def test_default_builder_reveals_nothing(hog_env) -> None:
    assert not hog_env.obs_builder.reveal.any_enabled


@pytest.mark.parametrize("seat,other", [("blue", "red"), ("red", "blue")])
def test_opponent_deck_and_hand_invisible(seat: str, other: str) -> None:
    """Changing only the OPPONENT's deck (entirely different cards, different hand) leaves this
    seat's observations unchanged while the opponent has revealed nothing (both seats WAIT)."""
    decks_a = {seat: list(HOG_26), other: list(STARTER_GIANT)}
    decks_b = {seat: list(HOG_26), other: list(HOG_CONTROL)}
    ea = _env(decks_a["blue"], decks_a["red"])
    eb = _env(decks_b["blue"], decks_b["red"])
    oa, _ = ea.reset(seed=4)
    ob, _ = eb.reset(seed=4)
    assert _same(oa[seat], ob[seat])
    assert not _same(oa[other], ob[other])  # sanity: the opponent's own view does differ
    for _ in range(30):
        oa, *_ = ea.step({"blue": 0, "red": 0})
        ob, *_ = eb.step({"blue": 0, "red": 0})
        assert _same(oa[seat], ob[seat])


def test_opponent_hand_order_invisible() -> None:
    """Same opponent deck, different card order (so a different hand and next card)."""
    red_a = list(HOG_26)
    red_b = list(HOG_26[4:]) + list(HOG_26[:4])
    ea, eb = _env(list(HOG_26), red_a), _env(list(HOG_26), red_b)
    oa, _ = ea.reset(seed=1)
    ob, _ = eb.reset(seed=1)
    assert ea.engine.state().players[1].hand != eb.engine.state().players[1].hand
    assert _same(oa["blue"], ob["blue"])


def test_revealing_builder_is_distinguishable() -> None:
    """A privileged builder changes the observation layout, so it cannot be used silently."""
    fair = _env(list(HOG_26), list(STARTER_GIANT))
    priv = _env(list(HOG_26), list(STARTER_GIANT), Reveal(enemy_hand=True, enemy_next_card=True))
    assert priv.obs_builder.reveal.any_enabled
    fo, _ = fair.reset(seed=0)
    po, _ = priv.reset(seed=0)
    assert fo["blue"]["vector"].shape != po["blue"]["vector"].shape or not _same(fo["blue"], po["blue"])


def test_reset_clears_match_memory() -> None:
    """§6.6: a reset starts from a clean public memory, so observations after reset with the
    same seed are identical whatever happened in the previous episode."""
    env = _env(list(HOG_26), list(STARTER_GIANT))
    first, _ = env.reset(seed=9)
    rng = np.random.default_rng(0)
    for _ in range(80):
        acts = {s: int(rng.choice(np.flatnonzero(first[s]["action_mask"]))) for s in ("blue", "red")}
        first, *_ = env.step(acts)
    again, _ = env.reset(seed=9)
    fresh, _ = _env(list(HOG_26), list(STARTER_GIANT)).reset(seed=9)
    assert _same(again["blue"], fresh["blue"]) and _same(again["red"], fresh["red"])
