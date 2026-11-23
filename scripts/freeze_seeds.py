"""Generate (once) and freeze a seed manifest. Refuses to overwrite an existing manifest.

    .\\.venv\\Scripts\\python.exe scripts\\freeze_seeds.py hog_mirror_v1 --generator-seed 20261123
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from royale_ai.benchmarks import seeds

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_id")
    ap.add_argument("--generator-seed", type=int, required=True)
    ap.add_argument("--validation", type=int, default=200)
    ap.add_argument("--test", type=int, default=500)
    a = ap.parse_args()
    path = ROOT / "configs" / "seeds" / f"{a.manifest_id}.yaml"
    if path.exists():
        print(f"{path} exists; frozen manifests are never regenerated. Make a new version.")
        return 1
    m = seeds.generate(a.manifest_id, a.generator_seed, a.validation, a.test)
    header = (
        f"# FROZEN seed manifest {a.manifest_id}. Do not edit: the sha256 is checked on load.\n"
        f"# Regenerate only as a NEW manifest id (scripts/freeze_seeds.py).\n"
    )
    path.write_text(header + yaml.safe_dump(m, sort_keys=False, width=100), encoding="utf-8")
    print(f"wrote {path} sha256={m['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
