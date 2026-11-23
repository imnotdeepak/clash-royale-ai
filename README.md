# clash-royale-ai

An offline reinforcement-learning agent for a Clash Royale–style card battler, trained entirely in simulation with [RoyaleGym](https://pypi.org/project/royalegym/) / RoyaleSim. It starts with a single deck (Hog 2.6). The long-term goal is a deck-agnostic agent.

This project never interacts with the live game.

## Status

**Phase P0: simulator and environment verification.** No neural-network work has started yet.

## Docs

- [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md): the single authoritative spec (phases, gates, contracts, training and evaluation protocol).

## Setup (Windows 11, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# CPU-only PyTorch (no NVIDIA GPU on the dev machine):
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\verify_install.py
```
