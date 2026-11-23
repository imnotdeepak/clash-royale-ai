"""Seat identifiers and the self-play vector layout (MASTER_PLAN §11.1).

Seats are lowercase ``blue`` / ``red`` everywhere in this project. The engine's integer team
index and the vector-env slot index are normalized ONLY here.

VERIFIED MAPPING (tests/test_seat_symmetry.py, royalegym 0.1.31 / royalesim 0.1.55):
  - team 0 = ``blue`` (king at small engine y), team 1 = ``red``
  - ClashSelfPlayVecEnv slot 2i = ``blue`` of game i, slot 2i+1 = ``red`` of game i
The mapping id below is recorded in every manifest as ``seat_mapping_verified``.
"""

from __future__ import annotations

from typing import Literal

Seat = Literal["blue", "red"]
SEATS: tuple[Seat, Seat] = ("blue", "red")
TEAM_OF: dict[str, int] = {"blue": 0, "red": 1}
SEAT_OF_TEAM: dict[int, Seat] = {0: "blue", 1: "red"}

SEAT_MAPPING_ID = "team0_blue__slot2i_blue_v1"
#: Versions the mapping was verified against at runtime. Bump the id if behaviour changes.
SEAT_MAPPING_VERIFIED_ON = {"royalegym": "0.1.31", "royalesim": "0.1.55"}


def validate_seat(seat: str) -> Seat:
    if seat not in TEAM_OF:
        raise ValueError(f"seat must be 'blue' or 'red', not {seat!r}")
    return seat  # type: ignore[return-value]


def other_seat(seat: str) -> Seat:
    return "red" if validate_seat(seat) == "blue" else "blue"


def team_of(seat: str) -> int:
    return TEAM_OF[validate_seat(seat)]


def seat_of_team(team: int) -> Seat:
    if team not in SEAT_OF_TEAM:
        raise ValueError(f"team must be 0 or 1, not {team!r}")
    return SEAT_OF_TEAM[team]


def vec_slot(game: int, seat: str) -> int:
    """Index of ``seat`` of game ``game`` in a ClashSelfPlayVecEnv batch."""
    return 2 * game + team_of(seat)


def slot_seat(slot: int) -> tuple[int, Seat]:
    """(game, seat) of a ClashSelfPlayVecEnv batch index."""
    if slot < 0:
        raise ValueError(f"slot must be >= 0, not {slot}")
    return slot // 2, SEAT_OF_TEAM[slot % 2]


def seat_mapping_record(versions: dict[str, str | None]) -> str:
    """The manifest value: mapping id plus the versions it was verified on vs. running."""
    ver = ",".join(f"{k}={v}" for k, v in sorted(SEAT_MAPPING_VERIFIED_ON.items()))
    run = ",".join(f"{k}={versions.get(k)}" for k in sorted(SEAT_MAPPING_VERIFIED_ON))
    return f"{SEAT_MAPPING_ID} (verified {ver}; running {run})"
