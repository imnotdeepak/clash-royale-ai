"""Action schema ``tiles_18x32_v1`` against the installed parser and engine (MASTER_PLAN §6.3, §6.6)."""

from __future__ import annotations

import numpy as np
import pytest
from royalegym import HalfTileActionParser, RustEngine, TileActionParser, make_env, to_own

from royale_ai.adapters.action_schema import TILES_18X32_V1 as V1
from royale_ai.decks import HOG_26

from .conftest import play_random


def test_schema_constants() -> None:
    assert V1.n_actions == 2305
    assert V1.encode(0, 0, 0) == 1
    assert V1.encode(3, 17, 31) == 2304
    assert V1.decode(0) is None


@pytest.mark.parametrize("slot", range(4))
@pytest.mark.parametrize("tx,ty", [(0, 0), (17, 0), (0, 31), (17, 31), (9, 15)])
def test_encode_decode_matches_installed_parser(hog_env, slot: int, tx: int, ty: int) -> None:
    parser = hog_env.action_parser
    assert isinstance(parser, TileActionParser)
    a = V1.encode(slot, tx, ty)
    assert a == parser.encode(slot, tx, ty) == 1 + slot * 576 + ty * 18 + tx
    assert V1.decode(a) == (slot, tx, ty) == tuple(parser.decode(a))


def test_every_action_round_trips() -> None:
    for a in range(1, V1.n_actions):
        slot, tx, ty = V1.decode(a)
        assert V1.encode(slot, tx, ty) == a
    with pytest.raises(ValueError):
        V1.decode(2305)


def test_installed_spaces(hog_env) -> None:
    assert hog_env.action_space("blue").n == hog_env.action_space("red").n == 2305
    assert hog_env.action_parser.config()["n_actions"] == 2305
    half = HalfTileActionParser()
    half.bind(RustEngine())
    assert half.space.n == 1 + 4 * 36 * 64  # documented alternative, unused


def test_wait_always_legal_and_masked_plays_always_accepted(hog_env) -> None:
    """A full random-legal match: WAIT legal every decision for both seats, and every
    executed masked play accepted by the engine (deploy_status OK)."""
    steps = plays = 0
    for obs, acts, (_, _, _, _, info) in play_random(hog_env, seed=11):
        steps += 1
        for s in ("blue", "red"):
            assert obs[s]["action_mask"][0] == 1
            if acts[s] != 0:
                plays += 1
                assert info[s]["deploy_status"] == 0, (s, acts[s], info[s]["deploy_status"])
    assert steps > 100 and plays > 20


#: KB-001 (royalesim 0.1.55): troops are accepted IN PLACE on these own-frame tiles, which lie
#: on the edge of the ENEMY king tower's 4x4 footprint (tiles x 7-10, y 27-30). The live game
#: refuses them; RoyaleGym's mask refuses them too. Same set for every troop card and, mirrored,
#: for both seats. Mitigation: only masked actions are ever executed (MASTER_PLAN §6.3).
KING_BOX_EDGE_OWN_FRAME = {
    "blue": {(10, 27), (10, 28), (10, 29), (7, 30), (8, 30), (9, 30), (10, 30)},
    "red": {(7, 27), (8, 27), (9, 27), (10, 27), (7, 28), (7, 29), (7, 30)},
}
KING_BOX_EDGE_OWN_FRAME_ANY = KING_BOX_EDGE_OWN_FRAME["blue"] | KING_BOX_EDGE_OWN_FRAME["red"]


def _resolved_action(env, cmd, team: int) -> int | None:
    """Execute ``cmd`` for one tick on a saved copy of the battle and return the v1 action of
    the tile the engine actually acted on (own frame), or None if refused."""
    blob = env.engine.save_state()
    results = env.engine.step([cmd], 1)
    res = next(r for r in results if r.team == team)
    env.engine.load_state(blob)
    if res.status != 0:
        return None
    x_own, y_own = to_own(env.engine.arena(), team, res.x, res.y)
    tx, ty = min(17, x_own // 18_000), min(31, y_own // 18_000)
    return V1.encode(cmd.hand_slot, tx, ty)


def test_mask_agrees_with_engine() -> None:
    """At sampled states of a random match, both seats:

    1. every masked-legal play is accepted by Engine.check_deploy (mask never too permissive);
    2. a masked-ILLEGAL tap the engine still accepts is one the engine RELOCATES (a troop or
       spell tapped past own territory is moved to the nearest legal point, as in the live
       client), and it lands on a tile the mask allows for the same slot -- so the mask
       drops only redundant taps and never a reachable placement.
    """
    env = make_env(deck=list(HOG_26), engine="rust")
    rng = np.random.default_rng(3)
    checked_legal = relocated = refused = 0
    for i, (_, _, res) in enumerate(play_random(env, seed=5)):
        if i % 25 != 0:
            continue
        obs = res[0]  # the observation that matches the engine's CURRENT state
        state = env.engine.state()
        for team, seat in ((0, "blue"), (1, "red")):
            mask = obs[seat]["action_mask"].astype(bool)
            legal = np.flatnonzero(mask[1:]) + 1
            illegal = np.flatnonzero(~mask[1:]) + 1
            for a in rng.choice(legal, size=min(20, len(legal)), replace=False) if len(legal) else []:
                cmd = env.action_parser.parse(int(a), state, team)
                assert cmd is not None and env.engine.check_deploy(cmd) == 0, (seat, a)
                checked_legal += 1
            for a in rng.choice(illegal, size=min(15, len(illegal)), replace=False):
                cmd = env.action_parser.parse(int(a), state, team)
                if cmd is None or env.engine.check_deploy(cmd) != 0:
                    refused += 1
                    continue
                landed = _resolved_action(env, cmd, team)
                assert landed is not None, (seat, a)
                if landed == a:
                    # Known engine bug KB-001 (below): in-place accept on the enemy king's box edge.
                    _, tx, ty = V1.decode(int(a))
                    assert (tx, ty) in KING_BOX_EDGE_OWN_FRAME[seat], f"{seat} tap {a} accepted in place yet masked"
                    continue
                assert mask[landed], f"{seat} tap {a} relocated to {landed}, which the mask refuses"
                relocated += 1
    assert checked_legal > 50 and refused > 50 and relocated > 0


def test_mask_shape_and_dtype(hog_env) -> None:
    obs, _ = hog_env.reset(seed=0)
    for s in ("blue", "red"):
        m = obs[s]["action_mask"]
        assert m.shape == (2305,) and set(np.unique(m)) <= {0, 1}


@pytest.mark.parametrize("seat", ["blue", "red"])
def test_known_bug_kb001_king_box_edge_is_masked(seat: str) -> None:
    """KB-001 pinned: the engine accepts troops in place on 7 tiles of the enemy king box;
    the mask must refuse every one of them. If the engine is fixed, the first assertion fails
    and this test (and the doc entry) should be retired."""
    from royalegym.protocol import DeployCommand, to_engine

    from royale_ai.scenarios.harness import make_board

    team = 0 if seat == "blue" else 1
    b = make_board(["Skeletons", "HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Fireball", "Log"])
    env = make_env(deck=list(HOG_26), engine="rust")
    obs, _ = env.reset(seed=0)
    for _ in range(12):  # past the deploy lockout, elixir to spare
        obs, *_ = env.step({"blue": 0, "red": 0})
    kinds = {c.card_id: c.card_kind for c in env.engine.cards()}
    hand = env.engine.state().players[team].hand
    troop_slots = [i for i, cid in enumerate(hand) if kinds[cid] == "TROOP"]
    assert troop_slots
    arena, base = b.engine.arena(), b.engine.save_state()
    accepted_in_place = set()
    for tx, ty in KING_BOX_EDGE_OWN_FRAME[seat]:
        x, y = to_engine(arena, team, tx * 18_000 + 9_000, ty * 18_000 + 9_000)
        b.engine.load_state(base)
        r = next(q for q in b.engine.step([DeployCommand(team, 0, x, y)], 1) if q.team == team)
        if r.status == 0 and (r.x, r.y) == (x, y):
            accepted_in_place.add((tx, ty))
        for slot in troop_slots:  # spells may target anywhere; troops may not stand here
            assert not obs[seat]["action_mask"][V1.encode(slot, tx, ty)], (seat, slot, tx, ty)
    assert accepted_in_place == KING_BOX_EDGE_OWN_FRAME[seat], "KB-001 behaviour changed; re-audit"
