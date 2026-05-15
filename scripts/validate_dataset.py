"""Validate collected LIBERO oracle demonstration datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_episode(path: Path) -> Dict:
    import h5py

    with h5py.File(path, "r") as h5:
        metadata = json.loads(h5.attrs.get("metadata", "{}"))
        obs_keys = list(h5["observations"].keys()) if "observations" in h5 else []
        actions = np.asarray(h5["actions"])
        initial = {}
        for key in obs_keys:
            data = np.asarray(h5["observations"][key])
            if data.size and data.ndim >= 2 and key.endswith("_pos"):
                initial[key] = data[0].reshape(-1).tolist()
        return {
            "path": str(path),
            "metadata": metadata,
            "success": bool(h5.attrs.get("success", False)),
            "obs_keys": obs_keys,
            "actions_shape": actions.shape,
            "initial": initial,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--expected-successes", type=int, default=100)
    parser.add_argument("--required-obs-key", action="append", default=["robot0_eef_pos", "robot0_proprio-state"])
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    episodes = [load_episode(path) for path in sorted(dataset_dir.glob("success_*.hdf5"))]
    errors: List[str] = []
    if len(episodes) != args.expected_successes:
        errors.append(f"expected {args.expected_successes} successful demos, found {len(episodes)}")
    seeds = [ep["metadata"].get("seed") for ep in episodes]
    if len(seeds) != len(set(seeds)):
        errors.append("duplicate seeds found")
    for ep in episodes:
        if not ep["success"] or not ep["metadata"].get("success", False):
            errors.append(f"success flag false: {ep['path']}")
        if len(ep["actions_shape"]) != 2 or ep["actions_shape"][1] != 7:
            errors.append(f"invalid action shape {ep['actions_shape']}: {ep['path']}")
        missing = [key for key in args.required_obs_key if key not in ep["obs_keys"]]
        if missing:
            errors.append(f"missing observation keys {missing}: {ep['path']}")

    initial_signatures = [json.dumps(ep["initial"], sort_keys=True) for ep in episodes]
    varied_initial_states = len(set(initial_signatures)) > 1 if episodes else False
    if args.expected_successes > 1 and not varied_initial_states:
        errors.append("initial states do not vary across successful episodes")

    report = {
        "dataset_dir": str(dataset_dir),
        "num_successes": len(episodes),
        "unique_seeds": len(set(seeds)),
        "varied_initial_states": varied_initial_states,
        "errors": errors,
        "valid": not errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

