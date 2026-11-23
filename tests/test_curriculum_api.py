"""Curriculum and scenario API compatibility (MASTER_PLAN §9.1, §12.2, §13 Task 2).

Behaviour, not imports: each mutator/parser is run in a real RustEngine episode and checked for
the property a curriculum depends on.
"""

from __future__ import annotations

import numpy as np
import pytest
from royalegym import (
    ClashParallelEnv,
    HalfTileActionParser,
    MidGameStateMutator,
    RustEngine,
    ScriptedBoardStateMutator,
    SnapshotStateMutator,
    TileActionParser,
    WeightedStateMutator,
)
from royalegym.protocol import SpawnSpec

from royale_ai.benchmarks.registry import SCENARIO_IDS, load_benchmarks
from royale_ai.decks import HOG_26

from .conftest import play_random


def _ids(names):
    m = {c.name: c.card_id for c in RustEngine().cards()}
    return [m[n] for n in names]


HOG = _ids(HOG_26)


def _env(mutator, **kw) -> ClashParallelEnv:
    return ClashParallelEnv(RustEngine(), state_mutator=mutator, **kw)


def _legal_and_fair(env, obs) -> None:
    assert not env.obs_builder.reveal.any_enabled
    for s in ("blue", "red"):
        assert obs[s]["action_mask"][0] == 1


def test_scripted_board_places_units_and_is_deterministic() -> None:
    hog = SpawnSpec(team=1, card_id=_ids(["HogRider"])[0], x=58_500, y=333_000)
    mut = ScriptedBoardStateMutator(spawns=[hog], decks=[HOG, HOG], elixir_milli=[4000, 7000], start_tick=200)
    env = _env(mut)
    obs, _ = env.reset(seed=1)
    _legal_and_fair(env, obs)
    st = env.engine.state()
    assert st.tick == 200 and [p.elixir_milli for p in st.players] == [4000, 7000]
    hogs = [e for e in st.entities if e.card_id == hog.card_id]
    assert len(hogs) == 1 and hogs[0].team == 1
    h1 = env.engine.state_hash()
    env.reset(seed=1)
    assert env.engine.state_hash() == h1


def test_midgame_mutator_respects_ranges() -> None:
    mut = MidGameStateMutator(
        tick_range=(1000, 2000), elixir_milli_range=(2000, 6000), max_tower_hp=(4824, 3052),
        tower_hp_percent=(40, 80), decks=[HOG, HOG],
    )
    env = _env(mut)
    for seed in range(8):
        obs, _ = env.reset(seed=seed)
        _legal_and_fair(env, obs)
        st = env.engine.state()
        assert 1000 <= st.tick <= 2000
        for p in st.players:
            assert 2000 <= p.elixir_milli <= 6000
            assert 4824 * 40 // 100 <= p.tower_hp[0] <= 4824 * 80 // 100
            assert all(3052 * 40 // 100 <= hp <= 3052 * 80 // 100 for hp in p.tower_hp[1:])


def test_snapshot_mutator_reproduces_saved_moment_with_history() -> None:
    src = _env(ScriptedBoardStateMutator(spawns=[], decks=[HOG, HOG]))
    gen = play_random(src, seed=4)
    for _ in range(60):
        _, _, (obs_at, *_) = next(gen)
    snap = src.snapshot(tag="mid_fight")
    env = _env(SnapshotStateMutator([snap]))
    obs, info = env.reset(seed=0)
    _legal_and_fair(env, obs)
    for s in ("blue", "red"):
        for k in obs[s]:
            np.testing.assert_array_equal(obs[s][k], obs_at[s][k], err_msg=f"{s} {k}")


def test_weighted_mutator_mixes_and_is_seed_deterministic() -> None:
    a = ScriptedBoardStateMutator(spawns=[], decks=[HOG, HOG], start_tick=111)
    b = ScriptedBoardStateMutator(spawns=[], decks=[HOG, HOG], start_tick=2222)
    env = _env(WeightedStateMutator([(a, 0.5), (b, 0.5)]))

    def ticks(seed0: int) -> list[int]:
        out = []
        for s in range(seed0, seed0 + 30):
            env.reset(seed=s)
            out.append(env.engine.state().tick)
        return out

    t = ticks(0)
    assert set(t) == {111, 2222}
    assert ticks(0) == t


@pytest.mark.parametrize("parser_cls,n", [(TileActionParser, 2305), (HalfTileActionParser, 9217)])
def test_action_parsers_run_full_episodes(parser_cls, n: int) -> None:
    env = _env(ScriptedBoardStateMutator(spawns=[], decks=[HOG, HOG]), action_parser=parser_cls())
    assert env.action_space("blue").n == n
    plays = 0
    for i, (obs, acts, (_, _, _, _, info)) in enumerate(play_random(env, seed=6)):
        assert obs["blue"]["action_mask"].shape == (n,)
        for s in ("blue", "red"):
            if acts[s] != 0:
                plays += 1
                assert info[s]["deploy_status"] == 0
        if i > 150:
            break
    assert plays > 10


def test_scenario_success_is_kept_separate_from_match_outcome() -> None:
    bench = load_benchmarks()
    assert bench["hog_tactical_scenarios"].raw["metric"] == "scenario_success_rate"
    assert not set(SCENARIO_IDS) & set(bench)


def test_missing_api_is_reported_clearly() -> None:
    import royalegym

    required = [
        "ScriptedBoardStateMutator", "MidGameStateMutator", "SnapshotStateMutator",
        "WeightedStateMutator", "HalfTileActionParser", "TileActionParser",
        "NoopOpponent", "RandomLegalOpponent", "FirstAffordableOpponent",
        "DefendOpponent", "PushOpponent", "PatientOpponent",
    ]
    missing = [n for n in required if not hasattr(royalegym, n)]
    assert missing == [], f"royalegym {royalegym.__version__} lacks {missing}"
