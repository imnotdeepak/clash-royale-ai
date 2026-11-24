"""The ``hog_cycle`` scripted Hog 2.6 pilot (MASTER_PLAN §5.6-5.7)."""

from __future__ import annotations

import numpy as np
import pytest
from royalegym import ClashParallelEnv, RustEngine, ScriptedBoardStateMutator
from royalegym.protocol import ShuffleMode, SpawnSpec

from royale_ai.adapters.action_schema import TILES_18X32_V1 as V1
from royale_ai.benchmarks.registry import RegistryError, load_deck_configs, validate_benchmarks
from royale_ai.benchmarks.runner import evaluate, play_game
from royale_ai.decks import HOG_26
from royale_ai.opponents.hog_cycle import HogCyclePilot
from royale_ai.opponents.public_view import PublicView, ViewReader
from royale_ai.opponents.registry import get_pilot

TILE = 18_000


def _ids(names):
    m = {c.name: c.card_id for c in RustEngine().cards()}
    return [m[n] for n in names]


def _env(blue_deck, red_deck=HOG_26, spawns=(), elixir=(10_000, 10_000), start_tick=100) -> ClashParallelEnv:
    mut = ScriptedBoardStateMutator(
        spawns=list(spawns), decks=[_ids(blue_deck), _ids(red_deck)], elixir_milli=list(elixir), start_tick=start_tick
    )
    mut.shuffle = ShuffleMode.NONE  # hand = first four cards, in order
    return ClashParallelEnv(RustEngine(), state_mutator=mut)


def _lead(*cards: str) -> list[str]:
    return list(cards) + [c for c in HOG_26 if c not in cards]


# ---------------------------------------------------------------------------- registry


def test_registered_as_hog_only_simple_scripted() -> None:
    p = get_pilot("hog_cycle")
    assert p.tier == "simple_scripted" and p.hog_only
    assert isinstance(p.build(), HogCyclePilot)


def test_cannot_pilot_non_hog_deck_in_registry() -> None:
    import yaml

    from royale_ai.benchmarks.registry import CONFIGS

    doc = yaml.safe_load((CONFIGS / "benchmarks.yaml").read_text(encoding="utf-8"))
    giant = next(b for b in doc["benchmarks"] if b["id"] == "hog_vs_giant_scripted")
    giant["pilot"] = "hog_cycle"
    with pytest.raises(RegistryError):
        validate_benchmarks(doc, load_deck_configs())


def test_unbound_pilot_refuses() -> None:
    with pytest.raises(RuntimeError, match="bound"):
        HogCyclePilot().act({}, np.ones(2305, dtype=np.int8))


# ---------------------------------------------------------------------------- fairness


def test_public_view_has_no_hidden_opponent_fields() -> None:
    hidden = {"enemy_hand", "enemy_next_card", "enemy_elixir", "enemy_deck"}
    assert not hidden & set(PublicView.__dataclass_fields__)


def test_decisions_independent_of_opponent_hidden_hand() -> None:
    """Same board, opponent deck in a different order (different hand / next card, both
    unrevealed while the opponent waits): the pilot makes identical decisions."""
    red_a = list(HOG_26)
    red_b = list(HOG_26[4:]) + list(HOG_26[:4])
    seqs = []
    for red in (red_a, red_b):
        env = _env(HOG_26, red)
        obs, _ = env.reset(seed=0)
        pilot = HogCyclePilot().bind(env, "blue")
        seq = []
        for _ in range(120):
            a = pilot.act(obs["blue"], obs["blue"]["action_mask"])
            seq.append(a)
            obs, _, term, trunc, _ = env.step({"blue": a, "red": 0})
            if term["blue"] or trunc["blue"]:
                break
        seqs.append(seq)
    assert seqs[0] == seqs[1] and any(a != 0 for a in seqs[0])


# ---------------------------------------------------------------------------- legality / determinism


@pytest.mark.parametrize("seat", ["blue", "red"])
def test_full_match_only_masked_actions(seat: str) -> None:
    row = play_game(seed=301, agent_seat=seat, agent_pilot="hog_cycle", opponent_pilot="random_legal")
    assert row.illegal_overrides == 0 and row.agent_plays > 10 and row.outcome != "not_applicable"


def test_deterministic() -> None:
    a = play_game(seed=302, agent_seat="blue", agent_pilot="hog_cycle", opponent_pilot="first_affordable")
    b = play_game(seed=302, agent_seat="blue", agent_pilot="hog_cycle", opponent_pilot="first_affordable")
    assert a == b


# ---------------------------------------------------------------------------- behaviours


def _first_play(env, pilot, seat="blue", steps=30):
    obs, _ = env.reset(seed=0)
    for _ in range(steps):
        a = pilot.act(obs[seat], obs[seat]["action_mask"])
        if a != 0:
            return a, pilot.last_rule
        obs, *_ = env.step({seat: a, "red" if seat == "blue" else "blue": 0})
    return 0, pilot.last_rule


def test_defends_hog_with_cannon_pull_and_saves_tower() -> None:
    hog = SpawnSpec(team=1, card_id=_ids(["HogRider"])[0], x=58_500, y=333_000)  # red, our left lane
    env = _env(_lead("Cannon"), spawns=[hog])
    pilot = HogCyclePilot().bind(env, "blue")
    a, rule = _first_play(env, pilot)
    slot, tx, ty = V1.decode(a)
    assert rule == "defend_cannon_pull" and env.engine.state().players[0].hand[slot] == _ids(["Cannon"])[0]
    assert 6 <= tx <= 8 and 9 <= ty <= 11
    obs, _ = env.reset(seed=0)
    for _ in range(40):
        act = pilot.act(obs["blue"], obs["blue"]["action_mask"])
        obs, *_ = env.step({"blue": act, "red": 0})
    assert env.engine.state().players[0].tower_hp[1] == 3052


def test_attacks_with_hog_at_bridge_when_idle_with_elixir() -> None:
    env = _env(_lead("HogRider"), elixir=(8000, 5000))
    a, rule = _first_play(env, HogCyclePilot().bind(env, "blue"))
    slot, tx, ty = V1.decode(a)
    assert rule == "attack_hog" and slot == 0 and ty == 14 and tx in (3, 14)


def test_logs_a_ground_swarm() -> None:
    skel = _ids(["Skeletons"])[0]
    spawns = [SpawnSpec(team=1, card_id=skel, x=x, y=200_000) for x in (150_000, 162_000, 174_000)]
    env = _env(_lead("Log"), red_deck=["Skeletons"] + [c for c in HOG_26 if c != "Skeletons"], spawns=spawns)
    a, rule = _first_play(env, HogCyclePilot().bind(env, "blue"))
    assert rule == "defend_log_swarm"


def test_never_stuck_at_full_elixir_with_non_cycle_hand() -> None:
    """Regression: hand Fireball/Log/Cannon/Musketeer at 10 elixir used to WAIT forever."""
    env = _env(_lead("Fireball", "Log", "Cannon", "Musketeer"), elixir=(10_000, 10_000))
    a, rule = _first_play(env, HogCyclePilot().bind(env, "blue"), steps=3)
    assert a != 0 and rule.startswith("anti_leak"), rule


@pytest.mark.parametrize("opponent", ["noop", "first_affordable"])
def test_elixir_leak_bounded_over_a_match(opponent: str) -> None:
    """At most 10% of decisions spent sitting on a full 10 elixir (it must keep cycling)."""
    from royalegym import make_env

    env = make_env(deck=list(HOG_26), engine="rust")
    obs, _ = env.reset(seed=303)
    pilot = HogCyclePilot().bind(env, "blue")
    opp = get_pilot(opponent).build()
    reader = ViewReader(env, "blue")
    rng = np.random.default_rng(0)
    full = total = 0
    while True:
        total += 1
        full += reader.read().elixir >= 10.0
        acts = {"blue": pilot.act(obs["blue"], obs["blue"]["action_mask"]),
                "red": int(opp.act(obs["red"], obs["red"]["action_mask"], rng))}
        obs, _, term, trunc, _ = env.step(acts)
        if term["blue"] or trunc["blue"]:
            break
    assert full / total <= 0.10, (full, total)


def test_beats_random_legal_on_both_seats() -> None:
    res = evaluate("hog_cycle", "random_legal", seeds=range(310, 313))
    t = res.tallies()
    assert t["blue"].wins == 3 and t["red"].wins == 3
