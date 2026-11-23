"""End-of-match rules (MASTER_PLAN §6.7), against the engine's AUTHORITATIVE game_over/winner.

Expected values were measured on royalesim 0.1.55 and agree with the ruleset the plan assumes:
3:00 regulation (3600 ticks), up to 2:00 sudden-death overtime (2400 ticks), king kill wins at
once, crown lead at regulation wins, overtime first crown wins, end-of-overtime tiebreak by the
client's tower-HP drain (lowest tower's side loses), exact ties are draws.
"""

from __future__ import annotations

import pytest
import royalegym as rg
from royalegym import ClashParallelEnv, RustEngine, ScriptedBoardStateMutator, StepLimitCondition
from royalegym.protocol import Winner

from royale_ai.adapters.episode import outcome_for, resolve_end
from royale_ai.decks import HOG_26
from royale_ai.scenarios.harness import BLUE, RED, make_board, tiles

H = list(HOG_26)
FULL = [4824, 3052, 3052]


def _finish(board, max_ticks: int):
    s = board.run_until(lambda s: s.game_over, max_ticks)
    assert s is not None, "match did not end"
    return s


def test_regulation_and_overtime_lengths() -> None:
    s = make_board(H).state
    assert s.tick_ms == 50 and s.regular_ticks == 3600 and s.overtime_ticks == 2400


def test_king_tower_destroyed_wins_immediately() -> None:
    b = make_board(H, tower_hp=[FULL, [1, 3052, 3052]], spawns=[(BLUE, "HogRider", tiles(9), tiles(27))])
    s = _finish(b, 400)
    assert s.winner == Winner.BLUE and not s.overtime and s.tick < 3600
    assert [p.crowns for p in s.players] == [3, 0]


def test_crown_lead_at_regulation_end_wins() -> None:
    b = make_board(H, start_tick=3500, tower_hp=[FULL, [4824, 0, 3052]])
    assert [p.crowns for p in b.state.players] == [1, 0]
    s = _finish(b, 200)
    assert s.tick == 3600 and s.winner == Winner.BLUE and not s.overtime


def test_equal_crowns_at_regulation_go_to_overtime() -> None:
    b = make_board(H, start_tick=3550)
    s = b.step(60)
    assert s.tick == 3610 and not s.game_over and s.overtime


def test_overtime_sudden_death_first_crown_wins() -> None:
    b = make_board(
        H, start_tick=3700, tower_hp=[FULL, [4824, 1, 3052]],
        spawns=[(BLUE, "HogRider", tiles(14.5), tiles(23))],
    )
    assert b.state.overtime
    s = _finish(b, 400)
    assert s.winner == Winner.BLUE and s.tick < 6000 and [p.crowns for p in s.players] == [1, 0]


@pytest.mark.parametrize("weaker", [BLUE, RED])
def test_end_of_overtime_tiebreak_lowest_tower_loses(weaker: int) -> None:
    hp = [[4824, 2000, 3052], [4824, 2000, 3052]]
    hp[weaker][1] = 1000
    b = make_board(H, start_tick=5950, tower_hp=hp)
    s = _finish(b, 600)
    assert s.tick > 6000  # decided by the drain after overtime's clock runs out
    assert s.winner == (RED if weaker == BLUE else BLUE)


def test_genuine_exact_tie_is_a_draw() -> None:
    s = _finish(make_board(H, start_tick=5950), 600)
    assert s.winner == Winner.DRAW
    assert s.players[0].tower_hp == s.players[1].tower_hp


def test_simultaneous_king_kills_are_a_draw() -> None:
    """Boundary event: mirrored Hogs destroy both kings on the same tick."""
    b = make_board(
        H, tower_hp=[[1, 3052, 3052], [1, 3052, 3052]],
        spawns=[(BLUE, "HogRider", tiles(9), tiles(27)), (RED, "HogRider", tiles(9), tiles(5))],
    )
    s = _finish(b, 400)
    assert s.winner == Winner.DRAW and s.players[0].tower_hp[0] == s.players[1].tower_hp[0] == 0


def test_crown_on_the_last_regulation_tick_counts() -> None:
    """Boundary tick: a princess tower destroyed exactly when regulation ends. Find the kill
    offset from tick 90, then shift the start so the kill lands on tick 3600."""
    spawn = [(BLUE, "HogRider", tiles(14.5), tiles(23))]
    hp = [FULL, [4824, 1, 3052]]
    probe = make_board(H, tower_hp=hp, spawns=spawn)
    s = probe.run_until(lambda s: s.players[1].tower_hp[1] == 0, 400)
    offset = s.tick - 90
    for kill_tick in (3599, 3600):
        b = make_board(H, start_tick=kill_tick - offset, tower_hp=hp, spawns=spawn)
        s = _finish(b, 400)
        crowns = [p.crowns for p in s.players]
        # Whatever tick it lands on, the engine's verdict agrees with the crowns it reports.
        assert crowns == [1, 0], (kill_tick, crowns)
        assert s.winner == Winner.BLUE and s.tick <= 3601, (kill_tick, s.tick, s.overtime)


def test_artificial_truncation_is_not_a_draw() -> None:
    env = ClashParallelEnv(
        RustEngine(),
        state_mutator=ScriptedBoardStateMutator(spawns=[], decks=None),
        truncation_cond=StepLimitCondition(5),
    )
    env.reset(seed=0)
    for i in range(5):
        _, _, term, trunc, info = env.step({"blue": 0, "red": 0})
    assert trunc["blue"] and trunc["red"] and not term["blue"] and not term["red"]
    end = resolve_end(term["blue"], trunc["blue"])
    assert end.bootstrap and not end.terminated
    assert outcome_for(info["blue"].get("winner", -1), 0, end.terminated) == "not_applicable"


def test_full_match_terminates_with_engine_outcome(hog_env) -> None:
    from .conftest import play_random

    for _, _, (_, _, term, trunc, info) in play_random(hog_env, seed=2):
        pass
    assert term["blue"] and not trunc["blue"]
    w = info["blue"]["winner"]
    assert info["blue"]["outcome"] == (0 if w == 2 else (1 if w == 0 else -1))
    assert outcome_for(w, 0, True) in ("win", "loss", "draw")


def test_both_flags_precedence() -> None:
    end = resolve_end(True, True)
    assert end.terminated and not end.truncated and end.anomaly and not end.bootstrap
    assert resolve_end(False, False).done is False
    with pytest.raises(ValueError):
        outcome_for(-1, 0, terminated=True)
    assert rg.Winner.DRAW == 2
