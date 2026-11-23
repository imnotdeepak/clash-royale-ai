"""P0 trainer audit (MASTER_PLAN §9.4, §13 Task 2): pin what the installed learners actually do
with discounts and episode ends, so a version bump that changes it fails here.

Findings (royalelearn 0.5.26, sb3-contrib 2.9.2):
- Both use ONE scalar gamma per update; neither takes per-transition elapsed time.
  => Path A (fixed 500 ms cadence) works with either; Path B needs custom code.
- royalelearn: terminated -> bootstrap 0, truncated -> V(final_obs), recursion cut at every
  episode end. Its default gamma is SCHEDULED 0.997 -> 0.999; we must pin a constant.
- MaskablePPO: timeout bootstrap is ``rewards += self.gamma * V(terminal_observation)``, only
  when ``infos[i]["TimeLimit.truncated"]`` is set; GAE masks with episode_starts.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from royale_ai.training import pbrs


def test_royalelearn_reference_gae_matches_our_returns_at_lambda_1() -> None:
    from royalelearn.learn.gae import reference_gae

    rng = np.random.default_rng(0)
    T = 40
    rewards = rng.normal(size=T)
    values = np.zeros(T + 1)  # zero critic: returns equal the discounted reward sums
    g = pbrs.step_gamma(pbrs.GAMMA_PER_SECOND_BASELINE, 0.5)
    for terminated in (True, False):
        term = np.zeros(T, bool)
        trunc = np.zeros(T, bool)
        (term if terminated else trunc)[-1] = True
        final_values = np.full(T, 3.0)
        _, ret = reference_gae(rewards, values, final_values, term, trunc, gamma=g, lam=1.0)
        ours = pbrs.returns_with_bootstrap(list(rewards), [g] * T, terminated=terminated, v_final=3.0)
        np.testing.assert_allclose(ret[:, 0], ours, rtol=1e-12)


def test_royalelearn_cuts_recursion_at_truncation() -> None:
    from royalelearn.learn.gae import reference_gae

    rewards = np.array([0.0, 0.0, 0.0, 10.0])
    values = np.zeros(5)
    trunc = np.array([False, True, False, False])
    adv, _ = reference_gae(rewards, values, np.array([0, 2.0, 0, 0]), np.zeros(4, bool), trunc, gamma=0.9, lam=0.95)
    assert adv[1, 0] == pytest.approx(0.9 * 2.0)  # bootstraps V(final_obs), not the next episode
    assert adv[0, 0] == pytest.approx(0.9 * 0.95 * adv[1, 0])


def test_royalelearn_default_gamma_is_scheduled_and_must_be_pinned() -> None:
    from royalelearn import config as cfg

    src = inspect.getsource(cfg)
    assert "gamma: ScheduleSpec = GeometricSpec(0.997, 0.999, 20_000_000)" in src
    assert hasattr(cfg, "ConstantSpec")


def test_royalelearn_gae_takes_scalar_gamma_only() -> None:
    from royalelearn.learn.gae import gae_recursion

    params = inspect.signature(gae_recursion).parameters
    assert "gamma" in params and not any(p in params for p in ("dt", "dt_seconds", "gammas"))


def test_maskableppo_fixed_gamma_timeout_bootstrap() -> None:
    sb3c = pytest.importorskip("sb3_contrib")
    from sb3_contrib.ppo_mask import ppo_mask

    src = inspect.getsource(ppo_mask)
    assert "rewards[idx] += self.gamma * terminal_value" in src
    assert 'infos[idx].get("TimeLimit.truncated", False)' in src
    assert 'infos[idx]["terminal_observation"]' in src
    assert sb3c.__version__ == "2.9.2"
