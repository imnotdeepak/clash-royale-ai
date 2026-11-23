"""Installation and runtime compatibility (MASTER_PLAN §12.2). CUDA is never required."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from royale_ai.decks import HOG_26


def test_python_version() -> None:
    assert sys.version_info >= (3, 12)


def test_torch_imports_without_requiring_cuda() -> None:
    import torch

    assert torch.tensor([1.0, 2.0]).sum().item() == 3.0
    # Reported, not required.
    assert isinstance(torch.cuda.is_available(), bool)


def test_royalegym_and_rust_engine_available() -> None:
    import royalegym

    assert royalegym.__version__
    assert royalegym.core_available(), royalegym.CORE_IMPORT_ERROR


def test_royalelearn_imports() -> None:
    import royalelearn  # noqa: F401


def test_hog_26_cards_in_catalogue() -> None:
    from royalegym import RustEngine

    names = {c.name for c in RustEngine().cards()}
    assert [c for c in HOG_26 if c not in names] == []


@pytest.fixture
def hog_env():
    from royalegym import make_env

    return make_env(deck=list(HOG_26), engine="rust")


def test_spaces_match_contract(hog_env) -> None:
    for seat in ("blue", "red"):
        assert hog_env.action_space(seat).n == 2305
        assert hog_env.observation_space(seat)["action_mask"].shape == (2305,)


def test_reset_and_step(hog_env) -> None:
    obs, info = hog_env.reset(seed=0)
    assert set(obs) == {"blue", "red"}
    for seat in ("blue", "red"):
        mask = obs[seat]["action_mask"]
        assert mask.shape == (2305,)
        assert mask[0] == 1, "WAIT (action 0) must always be legal"
    obs, rew, term, trunc, info = hog_env.step({"blue": 0, "red": 0})
    assert info["blue"]["tick"] > 0
    assert np.isfinite(rew["blue"]) and np.isfinite(rew["red"])
