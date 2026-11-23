"""The run manifest (MASTER_PLAN §11.1): written for every run and replay."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from royale_ai.adapters.action_schema import TILES_18X32_V1, ActionSchema
from royale_ai.adapters.seats import seat_mapping_record, validate_seat

MANIFEST_KEYS = (
    "experiment_id", "source_commit", "simulator_version", "royalegym_version",
    "royalelearn_version", "card_data_version", "config_hash", "observation_schema_version",
    "action_schema_version", "action_schema_hash", "decision_cadence_seconds",
    "gamma_per_second", "gamma_step_fixed", "gae_lambda_per_second", "rollout_discount_mode",
    "trainer_version", "reward_potential_version", "reward_alpha", "ruleset", "agent_deck",
    "opponent_deck", "opponent_pilot", "pilot_strength_tier", "seat", "seat_mapping_verified",
    "seed", "deck_split_version", "seed_manifest_hashes", "checkpoint_id",
    "evaluation_suite_version", "terminated", "truncated", "outcome",
)


def _ver(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def source_commit(repo: Path | None = None) -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_hash(config: Any) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def card_data_version(engine: Any) -> Any:
    """The engine's card-table stamp (hash + client vintage), kept structured."""
    stamp = getattr(engine, "card_table_stamp", None)
    stamp = stamp() if callable(stamp) else stamp
    if stamp is None:
        return "unknown"
    return dict(stamp) if isinstance(stamp, dict) else str(stamp)


def build_manifest(
    *,
    experiment_id: str,
    engine: Any,
    config: dict[str, Any],
    agent_deck: str,
    opponent_deck: str,
    opponent_pilot: str,
    pilot_strength_tier: str,
    seat: str,
    seed: int,
    observation_schema_version: str = "royalegym_spatial_default_v1",
    action_schema: ActionSchema = TILES_18X32_V1,
    decision_cadence_seconds: float = 0.5,
    gamma_per_second: float | None = None,
    gae_lambda_per_second: float | None = None,
    rollout_discount_mode: str | None = None,
    trainer_version: str | None = None,
    reward_potential_version: str | None = None,
    reward_alpha: float | None = None,
    deck_split_version: str | None = None,
    seed_manifest_hashes: dict[str, str] | None = None,
    checkpoint_id: str | None = None,
    evaluation_suite_version: str | None = None,
    terminated: bool | None = None,
    truncated: bool | None = None,
    outcome: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_seat(seat)
    versions = {"royalegym": _ver("royalegym"), "royalesim": _ver("royalesim")}
    gamma_step = (
        gamma_per_second**decision_cadence_seconds
        if gamma_per_second is not None and rollout_discount_mode == "fixed_step_stock_maskableppo"
        else None
    )
    m: dict[str, Any] = {
        "experiment_id": experiment_id,
        "source_commit": source_commit(),
        "simulator_version": versions["royalesim"],
        "royalegym_version": versions["royalegym"],
        "royalelearn_version": _ver("royalelearn"),
        "card_data_version": card_data_version(engine),
        "config_hash": config_hash(config),
        "observation_schema_version": observation_schema_version,
        "action_schema_version": action_schema.schema_id,
        "action_schema_hash": action_schema.hash,
        "decision_cadence_seconds": decision_cadence_seconds,
        "gamma_per_second": gamma_per_second,
        "gamma_step_fixed": gamma_step,
        "gae_lambda_per_second": gae_lambda_per_second,
        "rollout_discount_mode": rollout_discount_mode,
        "trainer_version": trainer_version,
        "reward_potential_version": reward_potential_version,
        "reward_alpha": reward_alpha,
        "ruleset": f"royalesim-{versions['royalesim']}-default-client",
        "agent_deck": agent_deck,
        "opponent_deck": opponent_deck,
        "opponent_pilot": opponent_pilot,
        "pilot_strength_tier": pilot_strength_tier,
        "seat": seat,
        "seat_mapping_verified": seat_mapping_record(versions),
        "seed": seed,
        "deck_split_version": deck_split_version,
        "seed_manifest_hashes": seed_manifest_hashes,
        "checkpoint_id": checkpoint_id,
        "evaluation_suite_version": evaluation_suite_version,
        "terminated": terminated,
        "truncated": truncated,
        "outcome": outcome,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": _ver("torch"),
            "engine_build": getattr(engine, "build_digest", lambda: None)()
            if callable(getattr(engine, "build_digest", None))
            else None,
        },
        "config": config,
    }
    if extra:
        m["extra"] = extra
    missing = [k for k in MANIFEST_KEYS if k not in m]
    assert not missing, missing
    return m


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path
