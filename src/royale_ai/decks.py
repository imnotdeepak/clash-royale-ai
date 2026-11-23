"""Canonical deck definitions (MASTER_PLAN §5), using the installed engine's card names.

Names are ``engine.cards()[i].name``. Display names differ: Ice Golem is ``IceGolemite``,
Ice Spirit is ``IceSpirits``, The Log is ``Log`` and Archers is ``Archer``.

A deck's identity is its canonical id: the sorted card multiset, so the same eight cards in
any order are the same deck (§10.3 "canonicalize deck composition before splitting").
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

DECK_SIZE = 8

#: Deck A: Hog 2.6, the agent deck and the mirror opponent (§5.2).
HOG_26: tuple[str, ...] = (
    "HogRider", "Musketeer", "Cannon", "IceGolemite",
    "IceSpirits", "Skeletons", "Fireball", "Log",
)

#: Deck B: Starter Giant (§5.3). Equal to RoyaleGym's STARTER_DECK as a set.
STARTER_GIANT: tuple[str, ...] = (
    "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer",
)

#: Deck C: Hog counter/control candidate, provisional (§5.4).
HOG_CONTROL: tuple[str, ...] = (
    "HogRider", "Knight", "Tesla", "Musketeer", "Skeletons", "Fireball", "Log", "IceSpirits",
)

#: Registry key -> cards. Keys match configs/decks/<key>.yaml.
DECKS: dict[str, tuple[str, ...]] = {
    "hog_26": HOG_26,
    "starter_giant": STARTER_GIANT,
    "hog_control": HOG_CONTROL,
}


def validate_deck(cards: Sequence[str]) -> None:
    if len(cards) != DECK_SIZE:
        raise ValueError(f"a deck has {DECK_SIZE} cards, got {len(cards)}: {list(cards)}")
    if len(set(cards)) != DECK_SIZE:
        dupes = sorted({c for c in cards if list(cards).count(c) > 1})
        raise ValueError(f"duplicate cards in deck: {dupes}")


def canonical_cards(cards: Sequence[str]) -> tuple[str, ...]:
    """The order-free form of a deck."""
    validate_deck(cards)
    return tuple(sorted(cards))


def canonical_deck_id(cards: Sequence[str]) -> str:
    """Stable id for a deck composition: ``deck_`` + 12 hex of sha256 over sorted names."""
    blob = "|".join(canonical_cards(cards)).encode()
    return "deck_" + hashlib.sha256(blob).hexdigest()[:12]
