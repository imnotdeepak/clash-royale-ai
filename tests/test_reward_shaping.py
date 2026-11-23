"""Discounted PBRS identities (MASTER_PLAN §9.2-9.3): telescoping on termination, the exact
residual on truncation, bootstrap masking, variable-duration discounts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from royale_ai.training import pbrs

ALPHA = 0.5


def _trajectory(rng: np.random.Generator, T: int, dts: list[float] | None = None):
    phis = list(rng.uniform(-1, 1, size=T + 1))
    dts = dts if dts is not None else [0.5] * T
    return phis, dts


@pytest.mark.parametrize("gps", pbrs.GAMMA_GRID)
@pytest.mark.parametrize("variable", [False, True])
def test_terminated_trajectory_telescopes_to_minus_alpha_phi0(gps: float, variable: bool) -> None:
    rng = np.random.default_rng(1)
    T = 400
    dts = list(rng.choice([0.5, 1.0, 2.0], size=T)) if variable else None
    phis, dts = _trajectory(rng, T, dts)
    gammas = [pbrs.transition_discount(gps, dt) for dt in dts]
    terms = [
        pbrs.shaping_term(ALPHA, gammas[t], phis[t], phis[t + 1], terminated=(t == T - 1))
        for t in range(T)
    ]
    total = pbrs.discounted_sum(terms, gammas)
    assert total == pytest.approx(-ALPHA * phis[0], abs=1e-9)
    assert total == pytest.approx(pbrs.shaping_residual(ALPHA, phis, gammas, True), abs=1e-9)


@pytest.mark.parametrize("gps", pbrs.GAMMA_GRID)
def test_truncated_trajectory_keeps_exact_residual(gps: float) -> None:
    rng = np.random.default_rng(2)
    T = 250
    phis, dts = _trajectory(rng, T, list(rng.choice([0.5, 1.0], size=T)))
    gammas = [pbrs.transition_discount(gps, dt) for dt in dts]
    terms = [pbrs.shaping_term(ALPHA, gammas[t], phis[t], phis[t + 1], False) for t in range(T)]
    total = pbrs.discounted_sum(terms, gammas)
    d_t = math.prod(gammas)
    assert total == pytest.approx(-ALPHA * phis[0] + ALPHA * d_t * phis[-1], abs=1e-9)
    assert total == pytest.approx(pbrs.shaping_residual(ALPHA, phis, gammas, False), abs=1e-9)


def test_phi_zeroed_only_on_true_termination() -> None:
    assert pbrs.shaping_term(1.0, 0.9, 0.2, 0.7, terminated=True) == pytest.approx(-0.2)
    assert pbrs.shaping_term(1.0, 0.9, 0.2, 0.7, terminated=False) == pytest.approx(0.9 * 0.7 - 0.2)


def test_truncation_gets_no_fabricated_outcome() -> None:
    assert pbrs.outcome_reward(None, terminated=False) == 0.0
    assert pbrs.outcome_reward("win", terminated=False) == 0.0
    assert pbrs.outcome_reward("win", terminated=True) == 1.0
    assert pbrs.outcome_reward("loss", terminated=True) == -1.0
    assert pbrs.outcome_reward("draw", terminated=True) == 0.0
    with pytest.raises(ValueError):
        pbrs.outcome_reward(None, terminated=True)


def test_shaped_reward_combines_outcome_and_shaping() -> None:
    tr = pbrs.Transition(phi=0.3, phi_next=-0.4, dt_seconds=0.5, terminated=True, truncated=False, outcome="win")
    assert pbrs.shaped_reward(tr, ALPHA, 0.998001) == pytest.approx(1.0 + ALPHA * (0.0 - 0.3))


def test_bootstrap_mask_uses_termination_not_truncation() -> None:
    rewards, gammas = [0.0, 0.0, 1.0], [0.999] * 3
    term = pbrs.returns_with_bootstrap(rewards, gammas, terminated=True, v_final=5.0)
    trunc = pbrs.returns_with_bootstrap(rewards, gammas, terminated=False, v_final=5.0)
    assert term[-1] == pytest.approx(1.0)
    assert trunc[-1] == pytest.approx(1.0 + 0.999 * 5.0)
    assert trunc[0] == pytest.approx(0.999**2 * (1.0 + 0.999 * 5.0))


def test_physical_time_discount_table() -> None:
    g = pbrs.GAMMA_PER_SECOND_BASELINE
    assert pbrs.transition_discount(g, 0.5) == pytest.approx(0.999, abs=1e-12)
    assert pbrs.transition_discount(g, 1.0) == pytest.approx(0.998001)
    assert pbrs.transition_discount(g, 0.25) == pytest.approx(0.9995, abs=1e-4)
    assert pbrs.transition_discount(g, 180) == pytest.approx(0.698, abs=1e-3)
    assert pbrs.transition_discount(g, 300) == pytest.approx(0.549, abs=1e-3)
    assert [round(pbrs.step_gamma(x, 0.5), 6) for x in pbrs.GAMMA_GRID] == [0.997, 0.999, 1.0]
    assert pbrs.lambda_per_second(0.99) == pytest.approx(0.9801)


def test_equal_physical_time_gives_equal_discount_across_cadences() -> None:
    g = pbrs.GAMMA_PER_SECOND_BASELINE
    d500 = math.prod([pbrs.step_gamma(g, 0.5)] * 120)
    d250 = math.prod([pbrs.step_gamma(g, 0.25)] * 240)
    assert d500 == pytest.approx(d250, rel=1e-12)


def test_tower_potential_bounded_and_antisymmetric() -> None:
    p = pbrs.tower_hp_potential_v1
    assert p([1, 1, 1], [1, 1, 1]) == 0.0
    assert p([1, 1, 1], [0, 0, 0]) == pytest.approx(1.0)
    assert p([0, 0, 0], [1, 1, 1]) == pytest.approx(-1.0)
    a, b = [0.9, 0.2, 1.0], [0.5, 1.0, 0.0]
    assert p(a, b) == pytest.approx(-p(b, a))
    with pytest.raises(ValueError):
        p([1.2, 1, 1], [1, 1, 1])
