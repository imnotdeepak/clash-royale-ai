from __future__ import annotations

import numpy as np
import pytest
from royalegym import make_env

from royale_ai.decks import HOG_26, STARTER_GIANT


@pytest.fixture
def hog_env():
    return make_env(deck=list(HOG_26), engine="rust")


@pytest.fixture
def hog_vs_giant_env():
    """Distinguishable decks: blue = Hog 2.6, red = Starter Giant (mechanics only, not a benchmark)."""
    return make_env(deck=[list(HOG_26), list(STARTER_GIANT)], engine="rust")


def play_random(env, seed: int, max_steps: int = 10_000):
    """Both seats random-legal from one rng; yields (obs, actions, step_result) per decision."""
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    for _ in range(max_steps):
        acts = {}
        for s in ("blue", "red"):
            legal = np.flatnonzero(obs[s]["action_mask"])
            acts[s] = int(rng.choice(legal))
        res = env.step(acts)
        yield obs, acts, res
        obs = res[0]
        if res[2]["blue"] or res[3]["blue"]:
            return
