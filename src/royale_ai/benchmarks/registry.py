"""Load and validate configs/benchmarks.yaml, configs/curriculum_hog.yaml and
configs/deck_splits.yaml (MASTER_PLAN §5.5-5.7, §9.1, §10.1, §10.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from royale_ai.decks import DECKS, canonical_deck_id
from royale_ai.opponents.registry import get_pilot

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs"

STATUSES = {"active", "pending_admission", "blocked", "not_before_phase"}
REQUIRED_G2 = {"hog_mirror_random_legal", "hog_mirror_noop"}
SCENARIO_IDS = (
    "legal_play_basics", "wait_under_pressure", "cannon_hog_pull", "ice_golem_kite",
    "ice_spirit_freeze_timing", "musketeer_support_defense", "giant_support_separation",
    "spell_tower_finisher", "hog_cycle_punish", "cycle_to_counter", "counterpush_conversion",
    "close_finish_lethal", "overtime_tiebreak",
)


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class Benchmark:
    id: str
    phase: str
    status: str
    raw: dict[str, Any]

    @property
    def pilot(self) -> str | None:
        return self.raw.get("pilot")

    @property
    def agent_deck(self) -> str | None:
        return self.raw.get("agent_deck")

    @property
    def opponent_deck(self) -> str | None:
        return self.raw.get("opponent_deck")


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_deck_configs(root: Path = CONFIGS) -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted((root / "decks").glob("*.yaml")):
        d = _yaml(p)
        key = d["deck_id"]
        if key != p.stem:
            raise RegistryError(f"{p.name}: deck_id {key!r} != file name")
        if key not in DECKS or tuple(d["cards"]) != DECKS[key]:
            raise RegistryError(f"{p.name}: cards differ from royale_ai.decks.DECKS[{key!r}]")
        out[key] = d
    return out


def validate_benchmarks(doc: dict[str, Any], decks: dict[str, dict[str, Any]]) -> dict[str, Benchmark]:
    out: dict[str, Benchmark] = {}
    for b in doc["benchmarks"]:
        bid = b["id"]
        if bid in out:
            raise RegistryError(f"duplicate benchmark {bid}")
        if b["status"] not in STATUSES:
            raise RegistryError(f"{bid}: unknown status {b['status']!r}")
        for field in ("agent_deck", "opponent_deck"):
            if field in b and b[field] not in decks:
                raise RegistryError(f"{bid}: unregistered {field} {b[field]!r}")
        if "pilot" in b:
            pilot = get_pilot(b["pilot"])  # KeyError on unregistered/retired ids
            opp = b.get("opponent_deck")
            if pilot.hog_only and opp is not None and opp != "hog_26":
                raise RegistryError(f"{bid}: Hog pilot {pilot.pilot_id} on non-Hog deck {opp}")
            pinned = decks.get(opp, {}).get("pinned_pilot") if opp else None
            if pinned is not None and b["pilot"] != pinned:
                raise RegistryError(f"{bid}: {opp} is pinned to {pinned}, not {b['pilot']}")
            eligible = decks.get(opp, {}).get("eligible_pilots") if opp else None
            if eligible is not None and b["pilot"] not in eligible:
                raise RegistryError(f"{bid}: pilot {b['pilot']} not eligible for {opp}")
            seats = b.get("agent_seats")
            if sorted(seats or []) != ["blue", "red"]:
                raise RegistryError(f"{bid}: agent_seats must be exactly [blue, red], got {seats}")
        if b["status"] == "active" and b.get("opponent_deck"):
            adm = decks[b["opponent_deck"]]["admission"]["status"]
            if adm.startswith("pending"):
                raise RegistryError(f"{bid}: active but {b['opponent_deck']} is {adm}")
        out[bid] = Benchmark(bid, b["phase"], b["status"], b)
    missing = REQUIRED_G2 - set(out)
    if missing:
        raise RegistryError(f"required G2 baselines missing: {sorted(missing)}")
    return out


def load_benchmarks(root: Path = CONFIGS) -> dict[str, Benchmark]:
    return validate_benchmarks(_yaml(root / "benchmarks.yaml"), load_deck_configs(root))


def load_scenarios(root: Path = CONFIGS) -> list[dict[str, Any]]:
    doc = _yaml(root / "curriculum_hog.yaml")
    ids = [s["id"] for s in doc["scenarios"]]
    if tuple(ids) != SCENARIO_IDS:
        raise RegistryError(f"scenario ids differ from MASTER_PLAN §9.1: {ids}")
    for s in doc["scenarios"]:
        for d in s["requires"]:
            if d not in DECKS:
                raise RegistryError(f"{s['id']}: unknown deck {d}")
    return doc["scenarios"]


# ---------------------------------------------------------------------------- deck splits


def validate_splits(doc: dict[str, Any]) -> dict[str, set[str]]:
    """Return pool -> set of canonical deck ids; raise on any leakage or mismatch."""
    pools: dict[str, set[str]] = {}
    for pool in ("train", "validation", "test"):
        ids: set[str] = set()
        for entry in doc["pools"][pool]:
            key, cid = entry["key"], entry["canonical_id"]
            if key not in DECKS:
                raise RegistryError(f"{pool}: unknown deck {key}")
            if canonical_deck_id(DECKS[key]) != cid:
                raise RegistryError(f"{pool}: {key} canonical id {cid} does not match its cards")
            if cid in ids:
                raise RegistryError(f"{pool}: {key} listed twice")
            ids.add(cid)
        pools[pool] = ids
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        both = pools[a] & pools[b]
        if both:
            raise RegistryError(f"deck(s) {sorted(both)} in both {a} and {b}")
    pending = {e["canonical_id"] for e in doc.get("pending_admission", [])}
    placed = set().union(*pools.values())
    if pending & placed:
        raise RegistryError(f"pending decks already placed in a pool: {sorted(pending & placed)}")
    if doc.get("frozen") and doc["split_version"].endswith("_draft"):
        raise RegistryError("a frozen split needs a final (non-draft) split_version")
    return pools


def load_splits(root: Path = CONFIGS) -> dict[str, set[str]]:
    return validate_splits(_yaml(root / "deck_splits.yaml"))


def allowed_in_training(cards: list[str] | tuple[str, ...], pools: dict[str, set[str]]) -> bool:
    """Curriculum generators call this: held-out (validation/test) decks are never trained on."""
    cid = canonical_deck_id(cards)
    return cid not in pools["validation"] and cid not in pools["test"]
