"""``hog_cycle``: a scripted Hog 2.6 pilot (MASTER_PLAN §5.6). Tier ``simple_scripted``.

A Hog pilot ONLY (§5.7): never use it for Starter Giant or the control deck.

It reads the board through ``PublicView`` (own hand/elixir + visible units + towers), never the
opponent's hand, deck or elixir, and only returns actions the mask allows. It is deterministic:
the same board gives the same move.

Rules, first match wins:
  1. DEFEND A BUILDING-TARGETER (Hog, Giant, ...) in or near our half: Cannon on the lane's
     pull tile; else Skeletons in front of the tower; else Ice Spirit on it; else Musketeer.
  2. DEFEND OTHER TROOPS in our half: Log a ground swarm (>= 2 small units); Fireball a valuable
     cluster; Musketeer against air or as general defence; Ice Golem to kite a heavy troop;
     else Skeletons on top of the attacker.
  3. SUPPORT a Hog we just sent: Ice Spirit behind it at the bridge.
  4. ATTACK with the Hog when elixir >= 7 (or 4 when the opponent just committed a big unit
     in their back), in the lane of the weaker enemy princess tower.
  5. DON'T LEAK: at >= 9.5 elixir play the Hog, else the cheapest cycle troop at the back.
  6. Otherwise WAIT.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from royale_ai.adapters.action_schema import TILES_18X32_V1 as V1
from royale_ai.opponents.public_view import PublicView, Unit, ViewReader

BUILDING_TARGETERS = {
    "HogRider", "Giant", "IceGolemite", "RoyalHogs", "Balloon", "GoblinGiant", "Golem",
    "RamRider", "BattleRam", "ElectroGiant", "LavaHound", "ElixirGolem", "WallBreakers",
}
CYCLE_TROOPS = ("IceSpirits", "Skeletons", "IceGolemite")

LEFT, RIGHT = "left", "right"
TOWER_X = {LEFT: 3, RIGHT: 14}
BRIDGE_X = {LEFT: 3, RIGHT: 14}
FRONT_ROW = 14  # last own-side row before the river (own frame)
CANNON_PULL = {LEFT: (7, 10), RIGHT: (10, 10)}


def lane_of(u: Unit) -> str:
    return LEFT if u.x < 9 else RIGHT


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class HogCyclePilot:
    pilot_id = "hog_cycle"
    tier = "simple_scripted"

    def __init__(self) -> None:
        self.reader: ViewReader | None = None
        self._support_until = -1  # tick until which a Hog push wants Ice Spirit support
        self._support_lane = LEFT
        self.last_rule = "wait"

    # ---------------------------------------------------------------- binding

    def bind(self, env, seat: str) -> HogCyclePilot:
        self.reader = ViewReader(env, seat)
        self._support_until = -1
        return self

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _slot(view: PublicView, card: str) -> int | None:
        return view.hand.index(card) if card in view.hand else None

    @staticmethod
    def _nearest_legal(mask: np.ndarray, slot: int, tx: float, ty: float, radius: int = 3) -> int | None:
        cx, cy = int(np.clip(round(tx - 0.5), 0, 17)), int(np.clip(round(ty - 0.5), 0, 31))
        best: tuple[float, int] | None = None
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < 18 and 0 <= y < 32:
                    a = V1.encode(slot, x, y)
                    if mask[a]:
                        d = dx * dx + dy * dy
                        if best is None or d < best[0]:
                            best = (d, a)
        return None if best is None else best[1]

    def _play(self, view: PublicView, mask: np.ndarray, card: str, tx: float, ty: float, rule: str) -> int | None:
        slot = self._slot(view, card)
        if slot is None:
            return None
        a = self._nearest_legal(mask, slot, tx, ty)
        if a is not None:
            self.last_rule = rule
        return a

    @staticmethod
    def _weaker_enemy_lane(view: PublicView) -> str:
        # Enemy tower slots are in THEIR frame: their LEFT (1) is on our RIGHT.
        # A destroyed tower (0 HP) counts as weakest: that lane leads to the King.
        on_our_left, on_our_right = view.enemy_towers[2], view.enemy_towers[1]
        return LEFT if on_our_left <= on_our_right else RIGHT

    # ---------------------------------------------------------------- policy

    def act(self, obs: dict[str, Any], mask: np.ndarray, rng: np.random.Generator | None = None) -> int:
        if self.reader is None:
            raise RuntimeError("HogCyclePilot must be bound to an env first: pilot.bind(env, seat)")
        mask = np.asarray(mask).astype(bool)
        view = self.reader.read()
        a = self._decide(view, mask)
        if a is None or not mask[a]:
            self.last_rule = "wait"
            return 0
        return int(a)

    def _decide(self, view: PublicView, mask: np.ndarray) -> int | None:
        enemies = view.enemies()

        # 1. building-targeters in or approaching our half
        pushers = sorted((u for u in enemies if u.card in BUILDING_TARGETERS and u.y < 19), key=lambda u: u.y)
        if pushers:
            h = pushers[0]
            lane = lane_of(h)
            if not any(u.mine and u.card == "Cannon" for u in view.units):
                a = self._play(view, mask, "Cannon", *CANNON_PULL[lane], rule="defend_cannon_pull")
                if a is not None:
                    return a
            if h.y < 13:
                a = self._play(view, mask, "Skeletons", TOWER_X[lane] + 0.5, 9.5, rule="defend_skeletons")
                if a is not None:
                    return a
            if h.y < 12:
                a = self._play(view, mask, "IceSpirits", h.x, h.y, rule="defend_ice_spirit")
                if a is not None:
                    return a
            if h.y < 14 and not any(u.mine and u.card == "Musketeer" for u in view.units):
                a = self._play(view, mask, "Musketeer", TOWER_X[lane] + 0.5, 4.5, rule="defend_musketeer")
                if a is not None:
                    return a

        # 2. other troops in our half
        invaders = [u for u in enemies if u.y < 15 and u.card not in BUILDING_TARGETERS]
        if invaders:
            ground = [u for u in invaders if not u.flying]
            centre = (float(np.mean([u.x for u in invaders])), float(np.mean([u.y for u in invaders])))
            small_ground = [u for u in ground if u.hp <= 400]
            if len(small_ground) >= 2:
                cx = float(np.mean([u.x for u in small_ground]))
                cy = float(np.min([u.y for u in small_ground]))
                a = self._play(view, mask, "Log", cx, max(1.5, cy - 1.5), rule="defend_log_swarm")
                if a is not None:
                    return a
            cluster = [u for u in invaders if _dist((u.x, u.y), centre) <= 2.5]
            if len(cluster) >= 3 or (len(cluster) >= 2 and sum(u.hp for u in cluster) >= 1400):
                a = self._play(view, mask, "Fireball", *centre, rule="defend_fireball")
                if a is not None:
                    return a
            if not any(u.mine and u.card == "Musketeer" for u in view.units):
                lane = LEFT if centre[0] < 9 else RIGHT
                a = self._play(view, mask, "Musketeer", TOWER_X[lane] + 0.5, 4.5, rule="defend_musketeer")
                if a is not None:
                    return a
            heavy = [u for u in ground if u.hp >= 1000]
            if heavy:
                t = min(heavy, key=lambda u: u.y)
                side = 2.0 if t.x < 9 else -2.0
                a = self._play(view, mask, "IceGolemite", t.x + side, max(1.5, t.y - 2), rule="defend_ice_golem_kite")
                if a is not None:
                    return a
            if ground:
                t = min(ground, key=lambda u: u.y)
                a = self._play(view, mask, "Skeletons", t.x, max(1.5, t.y - 1), rule="defend_skeletons")
                if a is not None:
                    return a

        # 3. support our Hog
        if view.tick <= self._support_until and any(u.mine and u.card == "HogRider" for u in view.units):
            a = self._play(view, mask, "IceSpirits", BRIDGE_X[self._support_lane] + 0.5, FRONT_ROW + 0.5,
                           rule="support_ice_spirit")
            if a is not None:
                self._support_until = -1
                return a

        # 4. attack
        enemy_heavy_back = any(u.hp >= 2000 and u.y > 24 for u in enemies)
        threshold = 4.0 if enemy_heavy_back else 7.0
        if view.elixir >= threshold and not invaders and not pushers:
            a = self._attack(view, mask, "attack_punish" if enemy_heavy_back else "attack_hog")
            if a is not None:
                return a

        # 5. don't leak
        if view.elixir >= 9.5:
            a = self._attack(view, mask, "anti_leak_hog")
            if a is not None:
                return a
            for card in CYCLE_TROOPS:
                a = self._play(view, mask, card, 8.5, 5.5, rule="anti_leak_cycle")
                if a is not None:
                    return a
            # Never get stuck at full elixir with a hand of non-cycle cards: whatever is in hand
            # gets spent, which also rotates the Hog back in.
            lane = self._weaker_enemy_lane(view)
            a = self._play(view, mask, "Musketeer", TOWER_X[lane] + 0.5, 4.5, rule="anti_leak_musketeer")
            if a is not None:
                return a
            tower = self._enemy_tower_tile(lane)
            for card in ("Log", "Fireball"):
                # The Log rolls forward ~10 tiles, so tap it short of the tower.
                ty = tower[1] - 4 if card == "Log" else tower[1]
                a = self._play(view, mask, card, tower[0], ty, rule="anti_leak_chip_" + card.lower())
                if a is not None:
                    return a
            a = self._play(view, mask, "Cannon", 8.5, 7.5, rule="anti_leak_cannon")
            if a is not None:
                return a
        self.last_rule = "wait"
        return None

    @staticmethod
    def _enemy_tower_tile(lane: str) -> tuple[float, float]:
        """Centre of the enemy princess tower in ``lane``, own frame (mirror of ours at y 6.5)."""
        return (TOWER_X[lane] + 0.5, 31 - 6.5 + 1)

    def _attack(self, view: PublicView, mask: np.ndarray, rule: str) -> int | None:
        lane = self._weaker_enemy_lane(view)
        a = self._play(view, mask, "HogRider", BRIDGE_X[lane] + 0.5, FRONT_ROW + 0.5, rule=rule)
        if a is not None:
            self._support_until = view.tick + 40  # next ~2 s
            self._support_lane = lane
        return a
