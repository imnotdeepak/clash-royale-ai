"""Canonical opponent pilots (MASTER_PLAN §5.6-5.7).

Only pilots registered here may appear in a benchmark. Each has a fixed strength tier.
``hog_cycle``, ``frozen_hog_vN``, ``current_selfplay`` and ``specialist_*`` are declared but
not yet implemented; asking for them raises ``PilotUnavailable`` rather than substituting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import royalegym as rg

PILOT_TIERS = ("passive", "random", "simple_scripted", "frozen_learned", "validated_specialist")


class PilotUnavailable(LookupError):
    pass


@dataclass(frozen=True)
class Pilot:
    pilot_id: str
    tier: str
    factory: Callable[[], object] | None
    hog_only: bool = False  # §5.7: Hog pilots must never pilot Deck B or C
    note: str = ""

    def build(self) -> object:
        if self.factory is None:
            raise PilotUnavailable(f"pilot {self.pilot_id!r} is not implemented yet")
        return self.factory()


def _hog_cycle() -> object:
    from royale_ai.opponents.hog_cycle import HogCyclePilot

    return HogCyclePilot()


PILOTS: dict[str, Pilot] = {
    "noop": Pilot("noop", "passive", rg.NoopOpponent),
    "random_legal": Pilot("random_legal", "random", rg.RandomLegalOpponent),
    "first_affordable": Pilot("first_affordable", "simple_scripted", rg.FirstAffordableOpponent),
    "defend": Pilot("defend", "simple_scripted", rg.DefendOpponent),
    "push": Pilot("push", "simple_scripted", rg.PushOpponent),
    "patient": Pilot("patient", "simple_scripted", rg.PatientOpponent),
    "hog_cycle": Pilot("hog_cycle", "simple_scripted", _hog_cycle, hog_only=True, note="custom, needs bind(env, seat)"),
}

RETIRED = {"cycle_fast", "basic_defender", "scripted_hog", "simple_hog"}


def get_pilot(pilot_id: str) -> Pilot:
    if pilot_id in RETIRED:
        raise KeyError(f"pilot id {pilot_id!r} is retired (§5.6)")
    if pilot_id in PILOTS:
        return PILOTS[pilot_id]
    if pilot_id.startswith("frozen_hog_v"):
        return Pilot(pilot_id, "frozen_learned", None, hog_only=True, note="checkpoint, P2")
    if pilot_id == "current_selfplay":
        return Pilot(pilot_id, "frozen_learned", None, hog_only=True, note="P2")
    if pilot_id.startswith("specialist_"):
        return Pilot(pilot_id, "validated_specialist", None, note="P3+")
    raise KeyError(f"unregistered pilot {pilot_id!r}")
