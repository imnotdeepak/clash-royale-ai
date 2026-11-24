"""Paired, seat-swapped match evaluation (MASTER_PLAN §10.4-10.5).

Every seed is played twice: agent on ``blue`` and agent on ``red``, same seed. Wins, losses and
draws are reported per seat and overall, with a Wilson interval on the decided-game win rate
and a score rate (draw = 1/2). Pilots are registry ids; pilots with ``bind(env, seat)`` (e.g.
``hog_cycle``) are bound to the live env before each game. Only masked actions are executed;
an unmasked choice is replaced by WAIT and counted (it must stay 0).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from royalegym import make_env

from royale_ai.adapters.episode import outcome_for, resolve_end
from royale_ai.adapters.seats import SEATS, other_seat, team_of
from royale_ai.decks import DECKS
from royale_ai.opponents.registry import get_pilot


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


@dataclass
class GameRow:
    seed: int
    agent_seat: str
    outcome: str  # win | loss | draw | not_applicable
    ticks: int
    agent_plays: int
    illegal_overrides: int


@dataclass
class Tally:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    undecided: int = 0

    def add(self, outcome: str) -> None:
        if outcome == "win":
            self.wins += 1
        elif outcome == "loss":
            self.losses += 1
        elif outcome == "draw":
            self.draws += 1
        else:
            self.undecided += 1

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws + self.undecided

    def summary(self) -> dict[str, Any]:
        decided = self.wins + self.losses
        lo, hi = wilson(self.wins, decided)
        played = self.wins + self.losses + self.draws
        return {
            **asdict(self),
            "games": self.games,
            "win_rate_decided": self.wins / decided if decided else None,
            "win_rate_ci95": [round(lo, 4), round(hi, 4)],
            "score_rate": (self.wins + 0.5 * self.draws) / played if played else None,
        }


@dataclass
class EvalResult:
    agent_pilot: str
    opponent_pilot: str
    agent_deck: str
    opponent_deck: str
    rows: list[GameRow] = field(default_factory=list)

    def tallies(self) -> dict[str, Tally]:
        out = {"overall": Tally(), "blue": Tally(), "red": Tally()}
        for r in self.rows:
            out["overall"].add(r.outcome)
            out[r.agent_seat].add(r.outcome)
        return out

    def summary(self) -> dict[str, Any]:
        t = self.tallies()
        return {
            "agent_pilot": self.agent_pilot,
            "opponent_pilot": self.opponent_pilot,
            "agent_deck": self.agent_deck,
            "opponent_deck": self.opponent_deck,
            "opponent_tier": get_pilot(self.opponent_pilot).tier,
            "overall": t["overall"].summary(),
            "by_seat": {s: t[s].summary() for s in SEATS},
            "illegal_overrides": sum(r.illegal_overrides for r in self.rows),
            "mean_ticks": float(np.mean([r.ticks for r in self.rows])) if self.rows else 0.0,
        }


def _build(pilot_id: str, env, seat: str):
    bot = get_pilot(pilot_id).build()
    if hasattr(bot, "bind"):
        bot.bind(env, seat)
    return bot


def play_game(
    seed: int, agent_seat: str, agent_pilot: str, opponent_pilot: str,
    agent_deck: str = "hog_26", opponent_deck: str = "hog_26",
    on_step: Callable[[dict, dict], None] | None = None,
) -> GameRow:
    opp_seat = other_seat(agent_seat)
    decks = {agent_seat: list(DECKS[agent_deck]), opp_seat: list(DECKS[opponent_deck])}
    env = make_env(deck=[decks["blue"], decks["red"]], engine="rust")
    obs, _ = env.reset(seed=seed)
    bots = {agent_seat: _build(agent_pilot, env, agent_seat), opp_seat: _build(opponent_pilot, env, opp_seat)}
    rng = np.random.default_rng(seed)
    illegal = plays = 0
    while True:
        acts = {}
        for s in SEATS:
            mask = obs[s]["action_mask"]
            a = int(bots[s].act(obs[s], mask, rng))
            if not mask[a]:
                a = 0
                if s == agent_seat:
                    illegal += 1
            if s == agent_seat and a != 0:
                plays += 1
            acts[s] = a
        if on_step is not None:
            on_step(obs, acts)
        obs, _, term, trunc, info = env.step(acts)
        if term["blue"] or trunc["blue"]:
            break
    end = resolve_end(term[agent_seat], trunc[agent_seat])
    outcome = outcome_for(info[agent_seat].get("winner", -1), team_of(agent_seat), end.terminated)
    return GameRow(seed, agent_seat, outcome, int(info["blue"]["tick"]), plays, illegal)


def evaluate(
    agent_pilot: str, opponent_pilot: str, seeds: Sequence[int],
    agent_deck: str = "hog_26", opponent_deck: str = "hog_26",
) -> EvalResult:
    res = EvalResult(agent_pilot, opponent_pilot, agent_deck, opponent_deck)
    for seed in seeds:
        for seat in SEATS:
            res.rows.append(play_game(seed, seat, agent_pilot, opponent_pilot, agent_deck, opponent_deck))
    return res
