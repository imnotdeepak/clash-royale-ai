"""Canonical deck definitions, using the installed engine's card names (engine.cards()[i].name).

Display names differ from engine names: Ice Golem is ``IceGolemite``, Ice Spirit is
``IceSpirits``, The Log is ``Log`` and Archers is ``Archer``.
"""

from __future__ import annotations

#: Deck A: Hog 2.6, the agent deck and the mirror opponent (MASTER_PLAN §5.2).
HOG_26: tuple[str, ...] = (
    "HogRider", "Musketeer", "IceGolemite", "IceSpirits",
    "Skeletons", "Cannon", "Fireball", "Log",
)
