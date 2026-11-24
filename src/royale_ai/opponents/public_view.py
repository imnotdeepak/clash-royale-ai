"""What one seat may legitimately know, extracted from the engine state (MASTER_PLAN §6.2).

Scripted pilots read the board through this view and NOTHING else, so a pilot cannot cheat:
  - own hand, own next card and own elixir (a player knows these);
  - every visible unit: team, card name, own-frame position, hp, flying, kind;
  - both sides' tower HP and the clock.
Never exposed: the opponent's hand, next card, deck order or exact elixir.
Everything is in the seat's OWN frame (own king at the bottom, small y), in tiles.
"""

from __future__ import annotations

from dataclasses import dataclass

from royalegym.protocol import EntityKind, to_own

from royale_ai.adapters.seats import team_of

TILE = 18_000


@dataclass(frozen=True)
class Unit:
    uid: int
    mine: bool
    card: str  # card name; "" for crown towers
    x: float  # own frame, tiles
    y: float
    hp: int
    flying: bool
    kind: int  # EntityKind
    deploying: bool


@dataclass(frozen=True)
class PublicView:
    tick: int
    regular_ticks: int
    overtime: bool
    elixir: float
    hand: tuple[str, ...]
    hand_costs: tuple[int, ...]
    next_card: str
    units: tuple[Unit, ...]  # troops and buildings, both sides
    own_towers: tuple[int, int, int]  # king, left, right HP (own frame slots)
    enemy_towers: tuple[int, int, int]

    def enemies(self) -> list[Unit]:
        return [u for u in self.units if not u.mine]

    def allies(self) -> list[Unit]:
        return [u for u in self.units if u.mine]


class ViewReader:
    """Builds a ``PublicView`` for one seat of a ClashParallelEnv."""

    def __init__(self, env, seat: str) -> None:
        self.env = env
        self.team = team_of(seat)
        self.arena = env.engine.arena()
        cards = env.engine.cards()
        self.names = {c.card_id: c.name for c in cards}
        self.cost = {c.card_id: c.elixir for c in cards}

    def read(self) -> PublicView:
        st = self.env.engine.state()
        me, them = st.players[self.team], st.players[1 - self.team]
        units = []
        for e in st.entities:
            if e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER):
                continue
            x, y = to_own(self.arena, self.team, e.x, e.y)
            units.append(Unit(
                uid=e.uid, mine=e.team == self.team, card=self.names.get(e.card_id, ""),
                x=x / TILE, y=y / TILE, hp=e.hp, flying=e.flying, kind=e.kind,
                deploying=e.deploy_ticks > 0,
            ))
        costs = tuple(me.hand_costs) if me.hand_costs else tuple(self.cost.get(c, 99) for c in me.hand)
        return PublicView(
            tick=st.tick, regular_ticks=st.regular_ticks, overtime=st.overtime,
            elixir=me.elixir_milli / 1000, hand=tuple(self.names.get(c, "") for c in me.hand),
            hand_costs=costs, next_card=self.names.get(me.next_card, ""),
            units=tuple(units), own_towers=tuple(me.tower_hp), enemy_towers=tuple(them.tower_hp),
        )
