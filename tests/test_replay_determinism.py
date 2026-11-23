"""Fixed seeds, snapshot/replay reproducibility and time increments (MASTER_PLAN §4, §13)."""

from __future__ import annotations

import numpy as np
import royalegym as rg
from royalegym import ReplayRecorder, RustEngine, load_trace, save_trace, verify_trace

from royale_ai.decks import HOG_26

from .conftest import play_random


def _hashes(seed: int, steps: int = 200) -> list[int]:
    env = rg.make_env(deck=list(HOG_26), engine="rust")
    out = []
    for i, _ in enumerate(play_random(env, seed=seed)):
        out.append(env.engine.state_hash())
        if i + 1 >= steps:
            break
    return out


def test_same_seed_same_battle() -> None:
    assert _hashes(21) == _hashes(21)


def test_different_seed_different_battle() -> None:
    assert _hashes(21) != _hashes(22)


def test_decision_advances_fixed_ticks(hog_env) -> None:
    """500 ms decisions at 50 ms ticks: every step advances exactly 10 engine ticks."""
    _, info = hog_env.reset(seed=0)
    assert hog_env.decision_ticks == 10 and hog_env.decision_ms == 500
    last = info["blue"]["tick"]
    for _ in range(50):
        _, _, _, _, info = hog_env.step({"blue": 0, "red": 0})
        assert info["blue"]["tick"] - last == 10 and info["red"]["tick"] == info["blue"]["tick"]
        last = info["blue"]["tick"]


def test_engine_save_load_round_trip() -> None:
    env = rg.make_env(deck=list(HOG_26), engine="rust")
    gen = play_random(env, seed=8)
    for _ in range(60):
        next(gen)
    blob = env.engine.save_state()
    h0 = env.engine.state_hash()
    def run() -> list[int]:
        hashes = []
        for _ in range(20):
            env.engine.step([], 7)
            hashes.append(env.engine.state_hash())
        return hashes

    a = run()
    env.engine.load_state(blob)
    assert env.engine.state_hash() == h0
    b = run()
    assert a == b and len(set(a)) > 1


def test_env_snapshot_restores_observations() -> None:
    env = rg.make_env(deck=list(HOG_26), engine="rust")
    gen = play_random(env, seed=13)
    for _ in range(80):
        _, _, (obs, *_) = next(gen)
    snap = env.snapshot()
    o1, _ = env.reset(seed=99, options={"snapshot": snap})
    for k in o1["blue"]:
        np.testing.assert_array_equal(o1["blue"][k], obs["blue"][k], err_msg=k)
        np.testing.assert_array_equal(o1["red"][k], obs["red"][k], err_msg=k)


def test_recorded_trace_replays_identically(tmp_path) -> None:
    env = rg.make_env(deck=list(HOG_26), engine="rust")
    env.recorder = ReplayRecorder()
    for _ in play_random(env, seed=31):
        pass
    trace = env.recorder.trace
    assert trace is not None
    path = save_trace(trace, tmp_path / "hog_mirror_seed31.msgpack")
    assert verify_trace(load_trace(path), RustEngine()) == []
