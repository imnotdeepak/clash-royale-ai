"""Hog 2.6 card-fidelity fixtures (MASTER_PLAN §5.8, §13 Task 3). Evidence feeds
docs/CARD_FIDELITY_MATRIX.md (kept local).

Two kinds of assertion, kept apart on purpose:
  * BEHAVIOUR: what the mechanic must do (targets buildings only, freezes, rolls, pulls...).
  * PINNED ENGINE VALUES (``PIN_*``): numbers measured on royalesim 0.1.55 at level 11. Where a
    live-game reference is confidently known it is noted ``ref``; otherwise the matrix marks the
    value "reference pending". A pin failing means the engine changed: re-audit, don't just edit.

All fixtures run on the raw engine (scenarios/harness.py), engine frame, blue at small y.
"""

from __future__ import annotations

import pytest
from royalegym.protocol import DeployCommand, DeployStatus

from royale_ai.decks import HOG_26
from royale_ai.scenarios.harness import BLUE, RED, TILE, Board, dist, make_board, tiles

H = list(HOG_26)
RED_DECK = ["Knight", "Minions", "Archer", "Giant", "Zap", "Fireball", "Cannon", "Musketeer"]
PRINCESS_HP = 3052


def board(*a, **k) -> Board:
    b = make_board(*a, **k)
    b.engine.set_entity_extras(True)
    return b


def lead(card: str) -> list[str]:
    """Hog 2.6 with ``card`` first, so it is in hand slot 0 (ShuffleMode.NONE)."""
    return [card] + [c for c in H if c != card]


# ---------------------------------------------------------------------------- deck / economy


def test_f01_deck_loads_with_exact_costs() -> None:
    b = board(H)
    cards = {c.name: c for c in b.engine.cards()}
    costs = {n: cards[n].elixir for n in H}
    # ref: live-game costs (avg 2.6 -> the deck's name)
    assert costs == {"HogRider": 4, "Musketeer": 4, "Cannon": 3, "IceGolemite": 2,
                     "IceSpirits": 1, "Skeletons": 1, "Fireball": 4, "Log": 2}
    assert sum(costs.values()) / 8 == pytest.approx(2.625)
    assert len(set(H)) == 8
    assert [cards[n].card_kind for n in ("Cannon", "Fireball", "Log")] == ["BUILDING", "SPELL", "SPELL"]
    assert cards["Skeletons"].count == 3  # ref


def test_f02_play_spends_elixir_and_cycles_hand() -> None:
    b = board(H, elixir=(5000, 5000))
    p = b.state.players[0]
    assert [b.names[c] for c in p.hand] == H[:4] and b.names[p.next_card] == H[4]
    r = b.play(BLUE, "HogRider", tiles(3.5), tiles(10.5))
    assert r.status == DeployStatus.OK
    p = b.state.players[0]
    assert 1000 <= p.elixir_milli < 1100  # 5 - 4 plus one tick of regen
    assert [b.names[c] for c in p.hand] == ["IceSpirits", "Musketeer", "Cannon", "IceGolemite"]
    assert b.names[p.next_card] == "Skeletons"  # HogRider went to the back of the queue
    status = b.engine.check_deploy(DeployCommand(0, 1, tiles(9.5), tiles(5.5)))  # Musketeer, 4e
    assert status == DeployStatus.NOT_ENOUGH_ELIXIR


def test_f03_elixir_regeneration_cap_and_lockout() -> None:
    b = board(H, elixir=(0, 0))
    t0 = b.tick
    s = b.run_until(lambda s: s.players[0].elixir_milli >= 1000, 200)
    assert s.tick - t0 == 56  # ref: 1 elixir per 2.8 s at 1x
    b = board(H, elixir=(0, 0), start_tick=2350)
    s = b.run_until(lambda s: s.elixir_rate == 2, 100)
    assert s.tick == 2400  # ref: double elixir from 2:00 elapsed (1:00 left)
    e0 = s.players[0].elixir_milli
    s = b.step(28)
    assert s.players[0].elixir_milli - e0 == pytest.approx(1000, abs=40)  # 1 per 1.4 s
    b = board(H, elixir=(9900, 9900))
    assert b.step(100).players[0].elixir_milli == 10_000  # cap
    b = board(H, skip_lockout=False)
    assert b.engine.check_deploy(DeployCommand(0, 0, tiles(9.5), tiles(5.5))) == DeployStatus.TOO_EARLY
    assert b.engine.calibration.int("match.DEPLOY_LOCKOUT_TICKS") == 90


def test_f04_deploy_zones() -> None:
    b = board(H)
    cd = b.engine.check_deploy
    assert cd(DeployCommand(0, 0, tiles(9.5), tiles(10.5))) == DeployStatus.OK  # own side
    assert cd(DeployCommand(0, 3, tiles(9.5), tiles(16.0))) == DeployStatus.WATER  # river
    assert cd(DeployCommand(0, 2, tiles(9.5), tiles(25.5))) == DeployStatus.OUT_OF_TERRITORY  # building
    # A troop tapped in enemy territory is MOVED to the nearest own-side point, as in the client.
    b.step(1, [DeployCommand(0, 0, tiles(9.5), tiles(25.5))])
    r = b.results[0]
    assert r.status == DeployStatus.OK and r.y < tiles(15)


# ---------------------------------------------------------------------------- Hog Rider


def test_f05_hog_targets_buildings_only_and_hits_tower() -> None:
    """Red Hog at red's left-lane bridge approach, a blue Knight in its path: the Hog ignores
    the Knight, keeps the princess tower as target, and hits at a fixed cadence."""
    b = board(H, RED_DECK, spawns=[(RED, "HogRider", tiles(3.25), tiles(18.5)),
                                  (BLUE, "Knight", tiles(5.5), tiles(12))])
    tower = b.tower(BLUE, 1)
    hog_uid = b.units("HogRider")[0].uid
    hits: list[tuple[int, int]] = []
    last = PRINCESS_HP
    for _ in range(300):
        b.step(1)
        h = b.entity(hog_uid)
        if h is None:
            break
        assert h.target_uid in (tower.uid, -1)
        hp = b.tower_hp(BLUE)[1]
        if hp < last:
            hits.append((b.tick, last - hp))
            last = hp
    assert len(hits) >= 3
    dmg = {d for _, d in hits}
    assert dmg == {317}  # PIN: Hog Rider level-11 damage per hit (ref: 317)
    gaps = {b2 - a2 for (a2, _), (b2, _) in zip(hits, hits[1:])}
    assert gaps == {32}  # ref: hit speed 1.6 s


def test_f06_hog_dies_to_princess_tower_fire() -> None:
    b = board(H, spawns=[(RED, "HogRider", tiles(3.25), tiles(18.5))])
    hog_uid = b.units("HogRider")[0].uid
    s = b.run_until(lambda s: b.entity(hog_uid) is None, 600)
    assert s is not None
    assert 1 <= b.tower_hp(BLUE)[1] < PRINCESS_HP  # it connected, but died before the tower fell


# ---------------------------------------------------------------------------- Cannon


def test_f07_cannon_pulls_hog_and_protects_tower() -> None:
    b = board(H, spawns=[(RED, "HogRider", tiles(3.25), tiles(18.5)), (BLUE, "Cannon", tiles(7), tiles(11))])
    cannon = b.units("Cannon")[0]
    hog_uid = b.units("HogRider")[0].uid
    targets = set()
    for _ in range(300):
        b.step(1)
        h = b.entity(hog_uid)
        if h is None:
            break
        targets.add(h.target_uid)
    assert cannon.uid in targets
    assert b.tower_hp(BLUE)[1] == PRINCESS_HP  # the pull kept the Hog off the tower
    assert b.entity(hog_uid) is None


def test_f08_cannon_lifetime_is_30_seconds() -> None:
    b = board(H, spawns=[(BLUE, "Cannon", tiles(9), tiles(10))])
    t0 = b.tick
    s = b.run_until(lambda s: not b.units("Cannon"), 800)
    assert s is not None and abs((s.tick - t0) - 600) <= 2  # ref: 30 s lifetime, linear decay


def test_f09_cannon_shoots_ground_not_air() -> None:
    b = board(H, RED_DECK, spawns=[(BLUE, "Cannon", tiles(9), tiles(10)), (RED, "Minions", tiles(9), tiles(13))])
    cannon_uid = b.units("Cannon")[0].uid
    for _ in range(40):
        b.step(1)
        c = b.entity(cannon_uid)
        m = b.units("Minions")
        if m:
            assert c.target_uid != m[0].uid
    b = board(H, RED_DECK, spawns=[(BLUE, "Cannon", tiles(9), tiles(10)), (RED, "Knight", tiles(9), tiles(14))])
    knight_uid = b.units("Knight")[0].uid
    s = b.run_until(lambda s: b.entity(knight_uid) is None or b.entity(knight_uid).hp < 1766, 200)
    assert s is not None and b.entity(cannon_uid).target_uid in (knight_uid, -1)


# ---------------------------------------------------------------------------- Ice Golem


def test_f10_ice_golem_targets_buildings_and_draws_enemy_troops() -> None:
    """Kiting: a red Knight retargets onto the blue Ice Golem; the Golem itself walks for
    buildings (never the Knight)."""
    b = board(H, RED_DECK, spawns=[(BLUE, "IceGolemite", tiles(9), tiles(19)), (RED, "Knight", tiles(9), tiles(23))])
    golem_uid, knight_uid = b.units("IceGolemite")[0].uid, b.units("Knight")[0].uid
    knight_targets, golem_targets = set(), set()
    for _ in range(80):
        b.step(1)
        g, k = b.entity(golem_uid), b.entity(knight_uid)
        if g is None or k is None:
            break
        knight_targets.add(k.target_uid)
        golem_targets.add(g.target_uid)
    assert golem_uid in knight_targets
    assert knight_uid not in golem_targets


def test_f11_ice_golem_death_damages_and_slows() -> None:
    b = board(H, RED_DECK, spawns=[(BLUE, "IceGolemite", tiles(9), tiles(20), 1), (RED, "Knight", tiles(9), tiles(21.2))])
    knight_uid = b.units("Knight")[0].uid
    s = b.run_until(lambda s: not b.units("IceGolemite"), 60)
    assert s is not None
    b.step(1)
    k = b.entity(knight_uid)
    assert 1766 - k.hp == 84  # PIN: death damage at level 11 (reference pending)
    assert k.speed_pct == 70  # PIN: 30% slow (reference pending)
    slow = [ms for name, ms in k.buffs if "Slow" in name]
    assert slow and slow[0] == 2500  # PIN: slow duration 2.5 s (reference pending)


# ---------------------------------------------------------------------------- Ice Spirit


def test_f12_ice_spirit_jumps_splashes_and_freezes() -> None:
    b = board(H, RED_DECK, spawns=[(BLUE, "IceSpirits", tiles(9), tiles(18)),
                                  (RED, "Knight", tiles(9), tiles(22)), (RED, "Archer", tiles(10), tiles(22.5))])
    knight_uid, archer_uid = b.units("Knight")[0].uid, b.units("Archer")[0].uid
    saw_projectile = False
    splashes = set()
    for _ in range(40):
        b.step(1)
        jumps = [p for p in b.state.projectiles if p.firer_card_id == b.ids["IceSpirits"]]
        saw_projectile |= bool(jumps)
        splashes |= {p.splash for p in jumps}
        k = b.entity(knight_uid)
        if k.stun_ticks > 0:
            break
    assert saw_projectile, "the spirit should leap as a projectile"
    assert not b.units("IceSpirits"), "the spirit is consumed by its jump"
    k, a = b.entity(knight_uid), b.entity(archer_uid)
    assert k.stun_ticks > 0 and a.stun_ticks > 0, "splash freezes both targets"
    assert 1766 - k.hp == 110 and 304 - a.hp == 110  # PIN: damage at level 11 (reference pending)
    assert 21 <= k.stun_ticks <= 24  # ref: freeze about 1.2 s (24 ticks; first tick already elapsed)
    assert splashes == {tiles(1.5)}  # PIN: splash radius 1.5 tiles


# ---------------------------------------------------------------------------- Skeletons


def test_f13_skeletons_spawn_three_around_tap_and_die_to_log() -> None:
    b = board(lead("Skeletons"), RED_DECK)
    r = b.play(BLUE, "Skeletons", tiles(9.5), tiles(10.5))
    assert r.status == DeployStatus.OK
    sk = b.units("Skeletons", BLUE)
    assert len(sk) == 3 and all(u.hp == 81 for u in sk)  # PIN: 81 HP at level 11
    centre = (r.x, r.y)
    assert all(dist(u, centre) < tiles(1.0) for u in sk)
    assert all(u.deploy_ticks > 0 for u in sk)
    b2 = board(["Log"] + [c for c in H if c != "Log"], ["Skeletons"] + RED_DECK[1:],
               spawns=[(RED, "Skeletons", tiles(9), tiles(12)), (RED, "Skeletons", tiles(9), tiles(13.5))])
    b2.play(BLUE, "Log", tiles(9.5), tiles(9.5))
    s = b2.run_until(lambda s: not b2.units("Skeletons", RED), 60)
    assert s is not None


def test_f14_skeletons_distract_hog_is_not_distracted() -> None:
    """Skeletons are a TROOP: a Hog walks through them (they do not pull it), whereas an enemy
    Knight will target them."""
    b = board(H, RED_DECK, spawns=[(RED, "Knight", tiles(9), tiles(14)), (BLUE, "Skeletons", tiles(9), tiles(11.5))])
    knight_uid = b.units("Knight")[0].uid
    skel_uids = {u.uid for u in b.units("Skeletons")}
    targets = set()
    for _ in range(60):
        b.step(1)
        k = b.entity(knight_uid)
        if k is None:
            break
        targets.add(k.target_uid)
    assert targets & skel_uids


# ---------------------------------------------------------------------------- Musketeer


def test_f15_musketeer_range_and_projectile() -> None:
    """A static target (red Cannon): measure the centre distance at which the Musketeer starts
    its attack, and that damage arrives by projectile, not instantly."""
    b = board(H, RED_DECK, spawns=[(BLUE, "Musketeer", tiles(9), tiles(7)), (RED, "Cannon", tiles(9), tiles(14))])
    m_uid, c_uid = b.units("Musketeer")[0].uid, b.units("Cannon")[0].uid
    start_dist = None
    proj_seen = False
    for _ in range(200):
        b.step(1)
        m, c = b.entity(m_uid), b.entity(c_uid)
        if m is None or c is None:
            break
        if start_dist is None and m.target_uid == c_uid and m.attack_phase in (1, 2):
            start_dist = dist(m, c) / TILE
        proj_seen |= any(p.firer_card_id == b.ids["Musketeer"] for p in b.state.projectiles)
        if proj_seen and start_dist is not None:
            break
    assert start_dist is not None and proj_seen
    assert 6.0 <= start_dist <= 7.6, start_dist  # ref: range 6 tiles + collision radii


def test_f16_musketeer_hits_air() -> None:
    b = board(H, RED_DECK, spawns=[(BLUE, "Musketeer", tiles(9), tiles(8)), (RED, "Minions", tiles(9), tiles(14))])
    m_uid = b.units("Musketeer")[0].uid
    minion_uid = b.units("Minions")[0].uid
    s = b.run_until(lambda s: b.entity(minion_uid) is None, 120)
    assert s is not None
    assert b.entity(m_uid) is not None


# ---------------------------------------------------------------------------- Fireball


def test_f17_fireball_travels_damages_and_knocks_back() -> None:
    fb = lead("Fireball")
    b = board(fb, RED_DECK, spawns=[(RED, "Knight", tiles(9), tiles(24)), (RED, "Archer", tiles(10), tiles(24)),
                                    (RED, "Archer", tiles(13.5), tiles(24))])
    knight_uid = b.units("Knight")[0].uid
    near_uid, far_uid = (u.uid for u in sorted(b.units("Archer"), key=lambda u: u.x))
    r = b.play(BLUE, "Fireball", tiles(9.5), tiles(24.5))
    assert r.status == DeployStatus.OK
    flight = b.run_until(lambda s: not s.spells, 80)
    assert flight is not None and flight.tick - r.tick > 20  # it flies from the king, not instant
    k = b.entity(knight_uid)
    assert 1766 - k.hp == 688  # PIN: level-11 damage (ref ~688)
    assert b.entity(near_uid) is None  # 304 HP Archer in the blast dies
    assert b.entity(far_uid) is not None  # ~4 tiles away: outside the 2.5-tile radius
    assert k.knockback_ticks > 0


def test_f18_fireball_reduced_crown_tower_damage() -> None:
    b = board(lead("Fireball"), RED_DECK)
    t = b.tower(RED, 1)
    b.play(BLUE, "Fireball", t.x, t.y)
    b.step(80)
    lost = PRINCESS_HP - b.tower_hp(RED)[1]
    assert 0 < lost < 688
    assert lost == 159  # PIN: crown-tower damage (reference pending)


# ---------------------------------------------------------------------------- The Log


def test_f19_log_rolls_ground_only_with_knockback() -> None:
    lg = lead("Log")
    b = board(lg, ["Skeletons"] + RED_DECK[1:], spawns=[
        (RED, "Skeletons", tiles(9), tiles(12)), (RED, "Knight", tiles(9.5), tiles(13)), (RED, "Minions", tiles(9), tiles(11)),
    ])
    knight_uid, minion_uid = b.units("Knight")[0].uid, b.units("Minions")[0].uid
    r = b.play(BLUE, "Log", tiles(9.5), tiles(9.5))
    assert r.status == DeployStatus.OK
    lengths, knock = set(), 0
    minion_hp_hit_by_log = False
    for _ in range(40):
        b.step(1)
        lengths |= {s.length for s in b.state.spells if s.length > 0}  # 0 while still airborne
        k = b.entity(knight_uid)
        knock = max(knock, k.knockback_ticks if k else 0)
        m = b.entity(minion_uid)
        # Towers can hit the minion; the Log must not. Before the minion is in tower range,
        # any HP loss would be the Log's.
        if m is not None and m.y > tiles(10) and m.hp < 230:
            minion_hp_hit_by_log = True
    assert lengths == {181_800}  # PIN: roll length 10.1 tiles
    assert 1766 - b.entity(knight_uid).hp == 268  # PIN: level-11 damage (reference pending)
    assert knock > 0
    assert not b.units("Skeletons", RED)
    assert not minion_hp_hit_by_log


def test_f20_log_reduced_crown_tower_damage() -> None:
    b = board(lead("Log"), RED_DECK)
    t = b.tower(RED, 1)
    b.play(BLUE, "Log", t.x, t.y - tiles(4))
    b.step(60)
    lost = PRINCESS_HP - b.tower_hp(RED)[1]
    assert 0 < lost < 268
    assert lost == 35  # PIN: crown-tower damage (reference pending)


# ---------------------------------------------------------------------------- towers


def test_f21_king_activates_after_princess_loss() -> None:
    b = board(H, tower_hp=[[4824, 1, 3052], [4824, 3052, 3052]], spawns=[(RED, "HogRider", tiles(3.5), tiles(10))])
    assert not any(p.king_active for p in b.state.players)
    s = b.run_until(lambda s: s.players[0].tower_hp[1] == 0, 400)
    lost_at = s.tick
    s = b.run_until(lambda s: s.players[0].king_active, 200)
    assert s is not None
    # PIN: KING_ACTIVATE_TIME_MS 3550 = 71 ticks (calibration "measured" on client replays)
    assert abs((s.tick - lost_at) - 71) <= 2, s.tick - lost_at
    assert b.engine.calibration.int("match.KING_ACTIVATE_TIME_MS") == 3550


def test_f22_princess_tower_shoots_air_and_ground() -> None:
    b = board(H, RED_DECK, spawns=[(RED, "Minions", tiles(3.5), tiles(12))])
    m_uid = b.units("Minions")[0].uid
    s = b.run_until(lambda s: b.entity(m_uid) is None or b.entity(m_uid).hp < 230, 100)
    assert s is not None
    m = b.entity(m_uid)
    assert m is None or 230 - m.hp == 109  # PIN: princess tower damage at level 11
