"""P0 Task 5: play one seeded Hog 2.6 match, save the replay, print the contract, write a manifest.

    .\\.venv\\Scripts\\python.exe scripts\\play_random.py [--seed 2000123] [--pilot first_affordable]
        [--agent-pilot random_legal] [--agent-seat blue] [--out runs/p0_replays]

The agent seat is played by ``--agent-pilot`` (a scripted stand-in until there is a policy). The
replay is re-verified against a fresh engine before the script reports success.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from royalegym import ReplayRecorder, RustEngine, make_env, save_trace, verify_trace

from royale_ai.adapters.episode import outcome_for, resolve_end
from royale_ai.adapters.seats import other_seat, team_of
from royale_ai.benchmarks import seeds as seed_manifests
from royale_ai.decks import HOG_26
from royale_ai.opponents.registry import get_pilot
from royale_ai.telemetry.manifest import build_manifest, write_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2_000_123)
    ap.add_argument("--pilot", default="first_affordable", help="opponent pilot id")
    ap.add_argument("--agent-pilot", default="random_legal", help="stand-in for the agent seat")
    ap.add_argument("--agent-seat", default="blue", choices=["blue", "red"])
    ap.add_argument("--out", default=str(ROOT / "runs" / "p0_replays"))
    a = ap.parse_args()

    agent_seat, opp_seat = a.agent_seat, other_seat(a.agent_seat)
    pilots = {agent_seat: get_pilot(a.agent_pilot), opp_seat: get_pilot(a.pilot)}
    bots = {s: p.build() for s, p in pilots.items()}

    env = make_env(deck=list(HOG_26), engine="rust")
    env.recorder = ReplayRecorder()
    obs, info = env.reset(seed=a.seed)
    rng = np.random.default_rng(a.seed)

    print("== contract")
    print("action_space", env.action_space("blue"))
    for k, v in obs["blue"].items():
        print(f"  obs[{k!r}] shape={v.shape} dtype={v.dtype}")
    print("reset info keys", sorted(info["blue"]))

    legal_counts: list[int] = []
    plays = Counter()
    illegal = 0
    steps = 0
    while True:
        acts = {}
        for s in ("blue", "red"):
            mask = obs[s]["action_mask"]
            act = int(bots[s].act(obs[s], mask, rng))
            if not mask[act]:
                illegal += 1
            acts[s] = act
            if s == agent_seat:
                legal_counts.append(int(mask.sum()))
            if act != 0:
                plays[s] += 1
        obs, rew, term, trunc, info = env.step(acts)
        steps += 1
        if term["blue"] or trunc["blue"]:
            break
    end = resolve_end(term[agent_seat], trunc[agent_seat])
    winner = info[agent_seat].get("winner", -1)
    outcome = outcome_for(winner, team_of(agent_seat), end.terminated)

    print("step info keys", sorted(info["blue"]))
    lc = np.array(legal_counts)
    print(f"legal actions per decision (agent): min={lc.min()} mean={lc.mean():.1f} max={lc.max()} "
          f"wait_only_fraction={(lc == 1).mean():.2f}")
    print(f"decisions={steps} ticks={info['blue']['tick']} plays={dict(plays)} illegal_executed={illegal}")
    print(f"terminated={end.terminated} truncated={end.truncated} winner={winner} agent_outcome={outcome}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"hog26_{agent_seat}_{a.agent_pilot}_vs_{a.pilot}_seed{a.seed}"
    trace_path = save_trace(env.recorder.trace, out / f"{stem}.msgpack")
    divergences = verify_trace(env.recorder.trace, RustEngine())
    print("replay", trace_path, "divergences:", len(divergences))

    manifest = build_manifest(
        experiment_id=f"p0_task5_{stem}",
        engine=env.engine,
        config={
            "script": "scripts/play_random.py",
            "decision_ms": env.decision_ms,
            "obs_builder": type(env.obs_builder).__name__,
            "reveal": env.obs_builder.reveal.as_dict(),
            "action_parser": env.action_parser.config(),
            "agent_pilot": a.agent_pilot,
        },
        agent_deck="hog_26",
        opponent_deck="hog_26",
        opponent_pilot=a.pilot,
        pilot_strength_tier=pilots[opp_seat].tier,
        seat=agent_seat,
        seed=a.seed,
        deck_split_version="deck_splits_v0_draft",
        seed_manifest_hashes={
            "hog_mirror_v1": seed_manifests.load(ROOT / "configs" / "seeds" / "hog_mirror_v1.yaml")["sha256"]
        },
        evaluation_suite_version="benchmarks_v1",
        terminated=end.terminated,
        truncated=end.truncated,
        outcome=outcome,
        extra={
            "replay": trace_path.name,
            "replay_divergences": len(divergences),
            "decisions": steps,
            "ticks": int(info["blue"]["tick"]),
            "illegal_executed": illegal,
        },
    )
    mpath = write_manifest(manifest, out / f"{stem}.manifest.yaml")
    print("manifest", mpath)
    print(json.dumps({k: manifest[k] for k in ("seat", "seat_mapping_verified", "action_schema_hash", "outcome")}, indent=1))
    ok = illegal == 0 and not divergences and end.terminated
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
