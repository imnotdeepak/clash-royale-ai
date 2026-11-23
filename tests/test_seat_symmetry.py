"""Seat identification, attribution and side-swapping (MASTER_PLAN §11.1, §12.2, §13 Task 2).

Verifies at RUNTIME the documentation claims: seats are ``blue``/``red``; vector slot 2i is
blue and 2i+1 red of game i; observations, actions, rewards and outcomes go to the right
seat; mirrored setups give each seat the same view; evaluation records seats correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
import royalegym as rg
from royalegym import (
    ClashParallelEnv,
    ClashSelfPlayVecEnv,
    RustEngine,
    ScriptedBoardStateMutator,
    TowerHPReward,
)
from royalegym.env import AGENTS
from royalegym.protocol import ShuffleMode, SpawnSpec, to_engine

from royale_ai.adapters import seats
from royale_ai.adapters.episode import final_info, final_observation
from royale_ai.adapters.action_schema import TILES_18X32_V1 as V1
from royale_ai.decks import HOG_26, STARTER_GIANT
from royale_ai.observations.symmetry import SymmetricObservations, own_king_box_actions

TILE = 18_000


def _ids(names):
    cards = {c.name: c.card_id for c in RustEngine().cards()}
    return [cards[n] for n in names]


def _scripted_env(spawns=(), elixir=(10_000, 10_000), tower_hp=None, decks=(HOG_26, STARTER_GIANT)):
    def make() -> ClashParallelEnv:
        return ClashParallelEnv(
            RustEngine(),
            reward_fn=TowerHPReward(),
            state_mutator=ScriptedBoardStateMutator(
                spawns=list(spawns),
                decks=[_ids(decks[0]), _ids(decks[1])],
                elixir_milli=list(elixir),
                tower_hp=tower_hp,
                start_tick=100,  # past the 90-tick opening deploy lockout
            ),
        )
    return make


# ---------------------------------------------------------------------------- identifiers


def test_installed_seat_identifiers_are_blue_red() -> None:
    assert tuple(AGENTS) == seats.SEATS == ("blue", "red")
    env = rg.make_env(deck=list(HOG_26), engine="rust")
    assert env.possible_agents == ["blue", "red"]
    obs, info = env.reset(seed=0)
    assert set(obs) == set(info) == {"blue", "red"}


def test_adapter_mapping_helpers() -> None:
    assert [seats.slot_seat(i) for i in range(4)] == [(0, "blue"), (0, "red"), (1, "blue"), (1, "red")]
    assert seats.vec_slot(3, "red") == 7 and seats.vec_slot(3, "blue") == 6
    assert seats.team_of("blue") == 0 and seats.seat_of_team(1) == "red"
    with pytest.raises(ValueError):
        seats.validate_seat("Blue")
    with pytest.raises(ValueError):
        seats.validate_seat(0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------- vector layout


def test_vec_slot_observations_belong_to_documented_seat() -> None:
    """Blue gets 10 elixir and red 0: only blue's slots may have legal plays."""
    vec = ClashSelfPlayVecEnv(3, _scripted_env(elixir=(10_000, 0)), viser=None)
    obs, _ = vec.reset(seed=0)
    masks = obs["action_mask"]
    for game in range(3):
        assert masks[seats.vec_slot(game, "blue")].sum() > 1
        assert masks[seats.vec_slot(game, "red")].sum() == 1  # WAIT only
        st = vec.envs[game].engine.state()
        assert st.players[0].deck == _ids(HOG_26) and st.players[1].deck == _ids(STARTER_GIANT)


@pytest.mark.parametrize("acting_slot", [0, 1, 2, 3])
def test_vec_slot_actions_applied_to_documented_seat(acting_slot: int) -> None:
    vec = ClashSelfPlayVecEnv(2, _scripted_env(), viser=None)
    obs, _ = vec.reset(seed=0)
    game, seat = seats.slot_seat(acting_slot)
    team = seats.team_of(seat)
    actions = np.zeros(4, dtype=np.int64)
    actions[acting_slot] = int(np.flatnonzero(obs["action_mask"][acting_slot])[1])
    vec.step(actions)
    for g, env in enumerate(vec.envs):
        st = env.engine.state()
        spent = [p.elixir_milli < 9_000 for p in st.players]
        expected = [g == game and t == team for t in (0, 1)]
        assert spent == expected, (acting_slot, g, spent)


def test_rewards_attributed_to_correct_seat() -> None:
    """A blue Hog on red's left princess tower: blue's slot gets positive tower reward,
    red's the negation, in every game of the vector."""
    a = RustEngine().arena()
    hog = SpawnSpec(team=0, card_id=_ids(["HogRider"])[0], x=261_000, y=430_000)
    vec = ClashSelfPlayVecEnv(2, _scripted_env(spawns=[hog]), viser=None)
    vec.reset(seed=0)
    total = np.zeros(4)
    for _ in range(12):
        _, r, *_ = vec.step(np.zeros(4, dtype=np.int64))
        total += r
    del a
    for g in range(2):
        b, rd = total[seats.vec_slot(g, "blue")], total[seats.vec_slot(g, "red")]
        assert b > 0 and rd < 0 and b == pytest.approx(-rd)


def test_outcome_and_final_obs_attributed_to_correct_seat() -> None:
    """Red's king at 1 HP with a blue Hog on it: game ends, blue wins, final_obs/final_info
    are delivered per slot and the next observation is already the new game's."""
    hog = SpawnSpec(team=0, card_id=_ids(["HogRider"])[0], x=162_000, y=486_000)
    vec = ClashSelfPlayVecEnv(
        1, _scripted_env(spawns=[hog], tower_hp=[[4824, 3052, 3052], [1, 3052, 3052]]), viser=None
    )
    vec.reset(seed=0)
    for _ in range(60):
        _, _, term, trunc, infos = vec.step(np.zeros(2, dtype=np.int64))
        if term.any() or trunc.any():
            break
    assert term.tolist() == [True, True] and trunc.tolist() == [False, False]
    fb, fr = final_info(infos, 0), final_info(infos, 1)
    assert fb["winner"] == fr["winner"] == 0
    assert fb["outcome"] == 1 and fr["outcome"] == -1
    for slot in (0, 1):
        fo = final_observation(infos, slot)
        assert set(fo) >= {"action_mask", "vector"}
    # auto-reset happened: the engine is back at the scenario's start tick, before the final one
    assert vec.envs[0].engine.state().tick < fb["tick"]


# ---------------------------------------------------------------------------- mirror symmetry


def _mirrored_pair():
    """Setup A: Hog(blue) vs Giant(red) with a blue Knight at P. Setup B: decks swapped and a
    red Knight at P rotated into red's frame: identical initial conditions up to the seat swap."""
    arena = RustEngine().arena()
    knight = _ids(["Knight"])[0]
    px, py = 4 * TILE + 9_000, 10 * TILE + 9_000
    rx, ry = to_engine(arena, 1, px, py)

    def env(decks, spawn):
        mut = ScriptedBoardStateMutator(spawns=[spawn], decks=[_ids(decks[0]), _ids(decks[1])], start_tick=100)
        mut.shuffle = ShuffleMode.NONE  # deal in deck order so both setups have identical hands
        return ClashParallelEnv(RustEngine(), state_mutator=mut)

    return (
        env((HOG_26, STARTER_GIANT), SpawnSpec(team=0, card_id=knight, x=px, y=py)),
        env((STARTER_GIANT, HOG_26), SpawnSpec(team=1, card_id=knight, x=rx, y=ry)),
    )


def _kb001_cells(zone_channel: int) -> tuple[set, set]:
    spatial = {(zone_channel, y, x) for y in range(27, 31) for x in range(7, 11)}
    actions = {int(a) for a in own_king_box_actions(range(4))}
    return spatial, actions


def test_mirrored_setups_symmetric_after_adapter_and_raw_diff_is_only_kb001() -> None:
    """Each seat must see exactly what its mirrored counterpart saw (while both wait).

    Raw RoyaleGym observations differ ONLY on the KB-001 king-box cells (pinned here, so an
    engine change is noticed); after ``SymmetricObservations`` they are identical."""
    ea, eb = _mirrored_pair()
    sa, sb = SymmetricObservations(ea), SymmetricObservations(eb)
    kb_spatial, kb_actions = _kb001_cells(sa.zone_channel)
    oa, _ = ea.reset(seed=3)
    ob, _ = eb.reset(seed=3)
    assert ea.engine.state().players[0].hand == eb.engine.state().players[1].hand
    raw_spatial, raw_actions = set(), set()
    for step in range(150):
        for a, b in (("blue", "red"), ("red", "blue")):
            raw_spatial |= {tuple(int(i) for i in d) for d in np.argwhere(oa[a]["spatial"] != ob[b]["spatial"])}
            raw_actions |= {int(i) for i in np.flatnonzero(oa[a]["action_mask"] != ob[b]["action_mask"])}
            np.testing.assert_array_equal(oa[a]["vector"], ob[b]["vector"], err_msg=f"step {step} vector")
            na, nb = sa.seat_obs(oa, a), sb.seat_obs(ob, b)
            for k in na:
                np.testing.assert_array_equal(na[k], nb[k], err_msg=f"step {step} {a} {k}")
        oa, *_ = ea.step({"blue": 0, "red": 0})
        ob, *_ = eb.step({"blue": 0, "red": 0})
    assert raw_spatial and raw_spatial <= kb_spatial, sorted(raw_spatial - kb_spatial)
    assert raw_actions and raw_actions <= kb_actions, sorted(raw_actions - kb_actions)


def test_adapter_never_masks_wait_or_spells() -> None:
    ea, _ = _mirrored_pair()
    sa = SymmetricObservations(ea)
    obs, _ = ea.reset(seed=0)
    for seat in ("blue", "red"):
        n = sa.seat_obs(obs, seat)
        assert n["action_mask"][0] == 1
        team = 0 if seat == "blue" else 1
        troop = set(sa.troop_slots(team))
        for slot in set(range(4)) - troop:
            sl = [V1.encode(slot, tx, ty) for ty in range(1, 5) for tx in range(7, 11)]
            np.testing.assert_array_equal(n["action_mask"][sl], obs[seat]["action_mask"][sl])


def test_side_swapped_evaluation_records_both_seats() -> None:
    res = rg.evaluate(
        rg.RandomLegalOpponent(),
        rg.NoopOpponent(),
        lambda: rg.make_env(deck=list(HOG_26), engine="rust"),
        games=4,
        seed=0,
    )
    assert sorted(s.seat for s in res.by_seat) == ["blue", "red"]
    assert all(s.games == 2 for s in res.by_seat)
    assert res.games == 4


def test_shuffle_mode_none_is_respected() -> None:
    """Guards the fixtures: ShuffleMode.NONE deals the first four cards in deck order."""
    e = RustEngine()
    e.reset(0, rg.MatchSetup(decks=[_ids(HOG_26), _ids(STARTER_GIANT)], shuffle=int(ShuffleMode.NONE)))
    assert e.state().players[0].hand == _ids(HOG_26)[:4]
