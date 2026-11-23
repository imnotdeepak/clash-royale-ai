"""Frozen seed manifests with paired blue/red seat assignments (MASTER_PLAN §13 Task 6).

A seed manifest lists battle seeds per split. Every seed is played twice, once with the agent
on ``blue`` and once on ``red`` (same seed, seats swapped), so seat advantage cancels in the
pair. Splits occupy disjoint integer ranges; training draws only from the train range.
The file stores a sha256 over its canonical content; loading verifies it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SPLIT_RANGES = {
    "train": (0, 1_000_000),
    "validation": (1_000_000, 2_000_000),
    "test": (2_000_000, 3_000_000),
}
SEAT_PAIRS = (("blue", "red"), ("red", "blue"))  # (agent seat, opponent seat)


def generate(manifest_id: str, generator_seed: int, n_validation: int, n_test: int) -> dict[str, Any]:
    rng = np.random.default_rng(generator_seed)
    out: dict[str, Any] = {
        "manifest_id": manifest_id,
        "generator_seed": generator_seed,
        "seat_pairs": [list(p) for p in SEAT_PAIRS],
        "train_range": list(SPLIT_RANGES["train"]),
    }
    for split, n in (("validation", n_validation), ("test", n_test)):
        lo, hi = SPLIT_RANGES[split]
        out[split] = sorted(int(s) for s in rng.choice(hi - lo, size=n, replace=False) + lo)
    out["sha256"] = content_hash(out)
    return out


def content_hash(m: dict[str, Any]) -> str:
    body = {k: v for k, v in m.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    m = yaml.safe_load(path.read_text(encoding="utf-8"))
    if m.get("sha256") != content_hash(m):
        raise ValueError(f"seed manifest {path} was altered: hash mismatch")
    validate(m)
    return m


def validate(m: dict[str, Any]) -> None:
    sets = {s: set(m[s]) for s in ("validation", "test")}
    if sets["validation"] & sets["test"]:
        raise ValueError("validation and test seeds overlap")
    for split, seeds in sets.items():
        lo, hi = SPLIT_RANGES[split]
        bad = [s for s in seeds if not lo <= s < hi]
        if bad:
            raise ValueError(f"{split} seeds outside {lo}..{hi}: {bad[:5]}")
    lo, hi = m["train_range"]
    for split, seeds in sets.items():
        if any(lo <= s < hi for s in seeds):
            raise ValueError(f"{split} seeds inside the train range")


def is_train_seed(seed: int) -> bool:
    lo, hi = SPLIT_RANGES["train"]
    return lo <= seed < hi


def paired_games(m: dict[str, Any], split: str) -> list[tuple[int, str]]:
    """(seed, agent_seat) for every game of ``split``: each seed on both seats."""
    return [(s, pair[0]) for s in m[split] for pair in m["seat_pairs"]]
