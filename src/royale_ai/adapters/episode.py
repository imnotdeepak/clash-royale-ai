"""Episode-end normalization at the RoyaleGym boundary (MASTER_PLAN §6.7, §9.2).

- ``terminated``: the match concluded (engine game over or a scenario's own success rule).
- ``truncated``: an external limit cut the episode. NOT a draw; bootstrap from the cut state.
- Both set: genuine termination wins, and the anomaly is reported.
- In auto-reset vector envs the bootstrap observation is ``infos["final_obs"][i]``, never the
  next match's first observation; a missing final observation is an error, not a fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

OUTCOMES = ("win", "loss", "draw", "not_applicable")


@dataclass(frozen=True)
class EpisodeEnd:
    terminated: bool
    truncated: bool
    anomaly: bool = False

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def bootstrap(self) -> bool:
        """Whether the value of the final state must be bootstrapped (true only on cut-offs)."""
        return self.truncated and not self.terminated


def resolve_end(terminated: bool, truncated: bool) -> EpisodeEnd:
    if terminated and truncated:
        log.warning("episode reported both terminated and truncated; termination wins")
        return EpisodeEnd(True, False, anomaly=True)
    return EpisodeEnd(bool(terminated), bool(truncated))


def outcome_for(winner: int, team: int, terminated: bool) -> str:
    """Outcome label for ``team`` from the engine's authoritative winner (0 blue, 1 red, 2 draw).

    Only a terminated episode has an outcome; a truncated one is ``not_applicable``.
    """
    if not terminated:
        return "not_applicable"
    if winner == 2:
        return "draw"
    if winner in (0, 1):
        return "win" if winner == team else "loss"
    raise ValueError(f"terminated episode without a winner (winner={winner})")


def final_observation(infos: dict[str, Any], index: int) -> Any:
    """The cut-off/terminal observation of vector slot ``index`` after an auto-reset.

    Accepts the gymnasium layout (``final_obs`` + ``_final_obs`` mask, what royalegym emits)
    and the SB3 layout (``terminal_observation`` per-env dicts). Raises KeyError if absent,
    because bootstrapping from the reset observation would be silently wrong.
    """
    if "final_obs" in infos:
        mask = infos.get("_final_obs")
        if mask is not None and not bool(mask[index]):
            raise KeyError(f"slot {index} did not end this step (_final_obs is False)")
        obs = infos["final_obs"][index]
        if obs is None:
            raise KeyError(f"final_obs[{index}] is None")
        return obs
    if "terminal_observation" in infos:
        return infos["terminal_observation"]
    raise KeyError("no final_obs / terminal_observation in infos; refusing to bootstrap")


def final_info(infos: dict[str, Any], index: int) -> dict[str, Any]:
    """Slot ``index``'s last-step info after an auto-reset.

    royalegym 0.1.31 packs ``final_info`` gymnasium-style: a dict of per-slot arrays, each key
    ``k`` with a ``_k`` presence mask. Returns a plain dict of the keys present for the slot.
    """
    fi = infos.get("final_info")
    if fi is None:
        raise KeyError("no final_info in infos")
    mask = infos.get("_final_info")
    if mask is not None and not bool(mask[index]):
        raise KeyError(f"slot {index} did not end this step (_final_info is False)")
    if isinstance(fi, dict):
        out = {}
        for k, v in fi.items():
            if k.startswith("_"):
                continue
            present = fi.get("_" + k)
            if present is None or bool(present[index]):
                item = v[index]
                out[k] = item.item() if hasattr(item, "item") else item
        return out
    return dict(fi[index])
