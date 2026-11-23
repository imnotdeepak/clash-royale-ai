"""Outcome reward, potential-based shaping and physical-time discounting (MASTER_PLAN §9.2-9.3).

    gamma_t    = gamma_per_second ** dt_seconds
    r_shaped_t = r_outcome_t + alpha * (gamma_t * Phi(s_next) - Phi(s_t))

Phi(s_next) := 0 only on TRUE termination; on truncation the observed Phi is kept and the
critic bootstraps. The discounted sum of the shaping terms telescopes to
``-alpha * Phi(s_0) + alpha * D_T * Phi(s_T)`` (``shaping_residual``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

GAMMA_PER_SECOND_BASELINE = 0.998001  # = 0.999 per 500 ms decision
GAMMA_GRID = (0.994009, 0.998001, 1.0)  # = {0.997, 0.999, 1.0} per 500 ms

OUTCOME_REWARD = {"win": 1.0, "loss": -1.0, "draw": 0.0}


def transition_discount(gamma_per_second: float, dt_seconds: float) -> float:
    if not 0.0 < gamma_per_second <= 1.0:
        raise ValueError(f"gamma_per_second must be in (0, 1], got {gamma_per_second}")
    if dt_seconds < 0:
        raise ValueError(f"dt_seconds must be >= 0, got {dt_seconds}")
    return gamma_per_second**dt_seconds


def step_gamma(gamma_per_second: float, decision_seconds: float) -> float:
    """The fixed per-step gamma for a stock fixed-cadence learner (Path A)."""
    return transition_discount(gamma_per_second, decision_seconds)


def lambda_per_second(lambda_500ms: float) -> float:
    return lambda_500ms**2


def outcome_reward(outcome: str | None, terminated: bool) -> float:
    """+1/-1/0 on true termination; truncated episodes get no fabricated terminal reward."""
    if not terminated:
        return 0.0
    if outcome not in OUTCOME_REWARD:
        raise ValueError(f"terminated episode needs an outcome in {list(OUTCOME_REWARD)}, got {outcome!r}")
    return OUTCOME_REWARD[outcome]


# ---------------------------------------------------------------------------- potentials


def tower_hp_potential_v1(own_frac: Sequence[float], enemy_frac: Sequence[float]) -> float:
    """``tower_hp_v1``: mean own tower-HP fraction minus mean enemy, bounded in [-1, 1].

    Inputs are the three crown towers' HP fractions (king, left, right) as each seat SEES them
    (public information: RoyaleGym's ``own_tower_hp_frac`` / ``enemy_tower_hp_frac``).
    """
    if len(own_frac) != 3 or len(enemy_frac) != 3:
        raise ValueError("expected three tower fractions per side")
    for f in (*own_frac, *enemy_frac):
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"tower fraction out of [0, 1]: {f}")
    return (sum(own_frac) - sum(enemy_frac)) / 3.0


POTENTIALS = {"tower_hp_v1": tower_hp_potential_v1}


# ---------------------------------------------------------------------------- shaping


@dataclass(frozen=True)
class Transition:
    phi: float  # Phi(s_t)
    phi_next: float  # Phi(s_{t+1}) as observed
    dt_seconds: float
    terminated: bool
    truncated: bool
    outcome: str | None = None  # only on terminated transitions


def shaping_term(alpha: float, gamma_t: float, phi: float, phi_next: float, terminated: bool) -> float:
    phi_next_eff = 0.0 if terminated else phi_next
    return alpha * (gamma_t * phi_next_eff - phi)


def shaped_reward(tr: Transition, alpha: float, gamma_per_second: float) -> float:
    g = transition_discount(gamma_per_second, tr.dt_seconds)
    r = outcome_reward(tr.outcome, tr.terminated)
    return r + shaping_term(alpha, g, tr.phi, tr.phi_next, tr.terminated)


def discounted_sum(values: Sequence[float], gammas: Sequence[float]) -> float:
    """sum_t D_t * values[t], D_0 = 1, D_{t+1} = D_t * gammas[t]."""
    total, d = 0.0, 1.0
    for v, g in zip(values, gammas, strict=True):
        total += d * v
        d *= g
    return total


def shaping_residual(alpha: float, phis: Sequence[float], gammas: Sequence[float], terminated: bool) -> float:
    """Closed form of the discounted shaping sum over a trajectory with potentials
    ``phis[0..T]`` and per-transition discounts ``gammas[0..T-1]``."""
    if len(phis) != len(gammas) + 1:
        raise ValueError("need T+1 potentials for T transitions")
    d_t = 1.0
    for g in gammas:
        d_t *= g
    phi_t = 0.0 if terminated else phis[-1]
    return -alpha * phis[0] + alpha * d_t * phi_t


def returns_with_bootstrap(
    rewards: Sequence[float], gammas: Sequence[float], terminated: bool, v_final: float
) -> list[float]:
    """Per-step discounted returns; the final state's value is added only on truncation."""
    g_next = 0.0 if terminated else v_final
    out = [0.0] * len(rewards)
    for t in range(len(rewards) - 1, -1, -1):
        g_next = rewards[t] + gammas[t] * g_next
        out[t] = g_next
    return out
