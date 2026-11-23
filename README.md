# clash-royale-ai

An offline reinforcement-learning agent for a Clash Royale–style card battler, trained entirely in simulation with [RoyaleGym](https://pypi.org/project/royalegym/) / RoyaleSim. It starts with a single deck (Hog 2.6). The long-term goal is a deck-agnostic agent.

This project never interacts with the live game.

## Status

**Phase P0: simulator and environment verification.** The P0 checklist is implemented. 163 tests cover installation, the action/mask contract, fair observations, seat mapping and symmetry, replay determinism, match-end rules, reward-shaping math, deck splits, the benchmark registry, the trainer audit, and 22 Hog 2.6 card-fidelity fixtures. No neural-network training has started yet.

## Docs

- The design spec (`docs/MASTER_PLAN.md`) is kept locally and is not published in this repo.

## Setup (Windows 11, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# CPU-only PyTorch (no NVIDIA GPU on the dev machine):
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\verify_install.py
```
