"""The project's action schema ``tiles_18x32_v1`` (MASTER_PLAN §6.3), frozen from the
installed ``TileActionParser``.

    0                                  -> WAIT
    1 + slot * (18 * 32) + ty * 18 + tx -> play hand ``slot`` at tile (tx, ty), own frame

``ty`` counts up from the ACTING seat's own king (own frame), so both seats share one head.
The schema hash covers everything a checkpoint's policy head depends on; loading across
different hashes is refused (``check_compatible``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

TILES_X = 18
TILES_Y = 32
HAND_SLOTS = 4
WAIT = 0


@dataclass(frozen=True)
class ActionSchema:
    schema_id: str
    n_actions: int
    tiles_x: int
    tiles_y: int
    hand_slots: int
    wait_actions: tuple[int, ...]
    frame: str
    parser: str

    @property
    def hash(self) -> str:
        blob = json.dumps(self.__dict__, sort_keys=True, default=list).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def encode(self, slot: int, tx: int, ty: int) -> int:
        if not (0 <= slot < self.hand_slots and 0 <= tx < self.tiles_x and 0 <= ty < self.tiles_y):
            raise ValueError(f"out of range: slot={slot} tx={tx} ty={ty}")
        return 1 + slot * (self.tiles_x * self.tiles_y) + ty * self.tiles_x + tx

    def decode(self, action: int) -> tuple[int, int, int] | None:
        """(slot, tx, ty), or None for WAIT."""
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} outside Discrete({self.n_actions})")
        if action in self.wait_actions:
            return None
        a = action - 1
        per_slot = self.tiles_x * self.tiles_y
        slot, rem = divmod(a, per_slot)
        ty, tx = divmod(rem, self.tiles_x)
        return slot, tx, ty


TILES_18X32_V1 = ActionSchema(
    schema_id="tiles_18x32_v1",
    n_actions=1 + HAND_SLOTS * TILES_X * TILES_Y,
    tiles_x=TILES_X,
    tiles_y=TILES_Y,
    hand_slots=HAND_SLOTS,
    wait_actions=(WAIT,),
    frame="own_seat_king_bottom",
    parser="royalegym.TileActionParser(ability_buttons=False)",
)

SCHEMAS: dict[str, ActionSchema] = {TILES_18X32_V1.schema_id: TILES_18X32_V1}


class SchemaMismatch(ValueError):
    pass


def check_compatible(saved_id: str, saved_hash: str, current: ActionSchema) -> None:
    """Refuse to load a policy head saved under another action schema (§6.4 rule 3)."""
    if saved_id != current.schema_id or saved_hash != current.hash:
        raise SchemaMismatch(
            f"checkpoint action schema {saved_id}/{saved_hash} != running "
            f"{current.schema_id}/{current.hash}; migrate and retrain the policy head instead"
        )
