"""A thin harness over the raw RoyaleSim engine for card-fidelity fixtures and rule tests.

Everything here is ENGINE frame (subtiles, 18 000 per tile; blue's king at the bottom,
small y). Fixtures build a board with ``MatchSetup`` spawns, then step the engine tick by
tick and read ``BattleState``. No observation builder or action parser is involved, so a
failure here is about the simulated mechanics, not about our adapters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from royalegym import MatchSetup, RustEngine
from royalegym.protocol import (
    BattleState,
    DeployCommand,
    DeployResult,
    EntityState,
    ShuffleMode,
    SpawnSpec,
)

TILE = 18_000
BLUE, RED = 0, 1
TICK_MS = 50
TICKS_PER_SECOND = 1000 // TICK_MS


def tiles(t: float) -> int:
    """``t`` tiles in subtiles."""
    return int(round(t * TILE))


def dist(a: EntityState | tuple[int, int], b: EntityState | tuple[int, int]) -> float:
    ax, ay = (a.x, a.y) if isinstance(a, EntityState) else a
    bx, by = (b.x, b.y) if isinstance(b, EntityState) else b
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


@dataclass
class Board:
    """One engine battle set up for a fixture."""

    engine: RustEngine
    ids: dict[str, int]
    names: dict[int, str]
    results: list[DeployResult] = field(default_factory=list)

    @property
    def state(self) -> BattleState:
        return self.engine.state()

    @property
    def tick(self) -> int:
        return self.state.tick

    def step(self, ticks: int = 1, commands: Sequence[DeployCommand] = ()) -> BattleState:
        self.results = list(self.engine.step(list(commands), ticks))
        return self.engine.state()

    def run_until(
        self, pred: Callable[[BattleState], bool], max_ticks: int, chunk: int = 1
    ) -> BattleState | None:
        """Step until ``pred(state)``; return that state, or None after ``max_ticks``."""
        for _ in range(0, max_ticks, chunk):
            s = self.step(chunk)
            if pred(s):
                return s
        return None

    def units(self, name: str | None = None, team: int | None = None) -> list[EntityState]:
        out = []
        for e in self.state.entities:
            if e.tower_slot != -1:
                continue
            if name is not None and self.names.get(e.card_id) != name:
                continue
            if team is not None and e.team != team:
                continue
            out.append(e)
        return out

    def tower(self, team: int, slot: int) -> EntityState | None:
        for e in self.state.entities:
            if e.team == team and e.tower_slot == slot:
                return e
        return None

    def tower_hp(self, team: int) -> list[int]:
        return list(self.state.players[team].tower_hp)

    def entity(self, uid: int) -> EntityState | None:
        for e in self.state.entities:
            if e.uid == uid:
                return e
        return None

    def play(self, team: int, card: str, x: int, y: int) -> DeployResult:
        """Play ``card`` from ``team``'s hand at engine-frame (x, y) this tick."""
        hand = self.state.players[team].hand
        cid = self.ids[card]
        if cid not in hand:
            raise AssertionError(f"{card} not in hand {[self.names.get(h) for h in hand]}")
        cmd = DeployCommand(team=team, hand_slot=hand.index(cid), x=x, y=y)
        self.step(1, [cmd])
        return next(r for r in self.results if r.team == team)


def make_board(
    blue_deck: Sequence[str],
    red_deck: Sequence[str] | None = None,
    *,
    spawns: Iterable[tuple[int, ...]] = (),
    seed: int = 0,
    elixir: tuple[int, int] | None = (10_000, 10_000),
    start_tick: int = 0,
    tower_hp: Sequence[Sequence[int]] | None = None,
    skip_lockout: bool = True,
) -> Board:
    """A battle with decks in the GIVEN order (hand = first four cards) and spawned units.

    ``spawns``: (team, card name, x, y[, hp]) in the engine frame; hp -1 or omitted = full.
    ``skip_lockout``: start past the opening deploy lockout so plays are accepted at once.
    """
    engine = RustEngine()
    cards = engine.cards()
    ids = {c.name: c.card_id for c in cards}
    names = {c.card_id: c.name for c in cards}
    red_deck = list(red_deck) if red_deck is not None else list(blue_deck)
    if skip_lockout and start_tick == 0:
        start_tick = engine.calibration.int("match.DEPLOY_LOCKOUT_TICKS")
    setup = MatchSetup(
        decks=[[ids[n] for n in blue_deck], [ids[n] for n in red_deck]],
        shuffle=int(ShuffleMode.NONE),
        start_tick=start_tick,
        elixir_milli=list(elixir) if elixir is not None else None,
        tower_hp=[list(r) for r in tower_hp] if tower_hp is not None else None,
        spawns=[
            SpawnSpec(team=sp[0], card_id=ids[sp[1]], x=sp[2], y=sp[3], hp=sp[4] if len(sp) > 4 else -1)
            for sp in spawns
        ],
    )
    engine.reset(seed, setup)
    return Board(engine=engine, ids=ids, names=names)
