"""P0 Task 1: verify the installation and report the runtime environment.

Usage (no venv activation needed):
    .\\.venv\\Scripts\\python.exe scripts\\verify_install.py [--json PATH]

Checks Python, PyTorch (CUDA is reported, never required), RoyaleGym/RoyaleSim/RoyaleLearn
imports and versions, that the Rust engine is available, that every Hog 2.6 card is in the
engine catalogue, and that one seeded Hog mirror match runs to completion. Exits non-zero
if any required check fails. See docs/MASTER_PLAN.md §13 Task 1.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import sys
import time
from typing import Any

PACKAGES = ("royalegym", "royalesim", "royalelearn", "royaleviser", "torch", "gymnasium", "numpy")


def pkg_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def torch_report() -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    return {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda,
        "device_name": torch.cuda.get_device_name(0) if cuda else None,
    }


def smoke_match(seed: int = 7) -> dict[str, Any]:
    import numpy as np
    from royalegym import FirstAffordableOpponent, RandomLegalOpponent, make_env

    from royale_ai.decks import HOG_26

    env = make_env(deck=list(HOG_26), engine="rust")
    obs, _ = env.reset(seed=seed)
    players = {"blue": FirstAffordableOpponent(), "red": RandomLegalOpponent()}
    rng = np.random.default_rng(seed)
    steps = 0
    t0 = time.perf_counter()
    while True:
        actions = {s: int(p.act(obs[s], obs[s]["action_mask"], rng)) for s, p in players.items()}
        obs, _, term, trunc, info = env.step(actions)
        steps += 1
        if term["blue"] or trunc["blue"]:
            break
    elapsed = time.perf_counter() - t0
    return {
        "seed": seed,
        "decisions": steps,
        "ticks": int(info["blue"]["tick"]),
        "terminated": bool(term["blue"]),
        "truncated": bool(trunc["blue"]),
        "blue_outcome": info["blue"].get("outcome"),
        "decisions_per_second": round(steps / elapsed, 1),
        "action_space": str(env.action_space("blue")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="also write the report to this path")
    args = ap.parse_args()

    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: pkg_version(p) for p in PACKAGES},
    }
    failures: list[str] = []

    if sys.version_info < (3, 12):
        failures.append(f"Python >= 3.12 required, found {report['python']}")

    try:
        report["torch"] = torch_report()
    except Exception as e:  # noqa: BLE001 - report every failure, keep going
        failures.append(f"torch: {e!r}")

    try:
        import royalegym

        report["rust_engine_available"] = bool(royalegym.core_available())
        if not report["rust_engine_available"]:
            failures.append(f"RoyaleSim Rust engine unavailable: {royalegym.CORE_IMPORT_ERROR!r}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"royalegym import: {e!r}")

    try:
        import royalelearn  # noqa: F401
    except Exception as e:  # noqa: BLE001
        failures.append(f"royalelearn import: {e!r}")

    if not failures:
        try:
            from royalegym import RustEngine

            from royale_ai.decks import HOG_26

            names = {c.name for c in RustEngine().cards()}
            missing = [c for c in HOG_26 if c not in names]
            report["hog_26_cards_present"] = not missing
            if missing:
                failures.append(f"Hog 2.6 cards missing from engine catalogue: {missing}")
            else:
                report["smoke_match"] = smoke_match()
                if report["smoke_match"]["action_space"] != "Discrete(2305)":
                    failures.append(f"unexpected action space {report['smoke_match']['action_space']}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"smoke match: {e!r}")

    report["failures"] = failures
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print("\nRESULT:", "PASS" if not failures else f"FAIL ({len(failures)} problem(s))")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
